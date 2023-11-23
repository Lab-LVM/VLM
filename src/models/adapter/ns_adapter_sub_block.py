import abc

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def fuse_bn(conv, bn, scale=1):
    kernel = conv.weight if not isinstance(conv, torch.Tensor) else conv
    running_mean = bn.running_mean * scale
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias * scale
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


def fuse_fc(fc_layers, scale=1):
    w, b = 0, 0
    for fc in fc_layers:
        w += fc.weight.data
        b += fc.bias.data * scale
    return w, b


def merge_1x1_kxk(k1, b1, k2, b2, groups=1):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) * k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append(
                (k2_slice * b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = torch.cat(k_slices, dim=0), torch.cat(b_slices)
    return k, b_hat + b2


def avg_to_kernel(channels, kernel_size, groups):
    kernel_size = kernel_size[0]
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k


def get_equivalent_kernel_bias(convbn, scale):
    eq_k, eq_b = 0, 0
    for i in range(len(convbn)):
        k, b = fuse_bn(convbn[i][0], convbn[i][1], scale)
        eq_k += k
        eq_b += b
    return eq_k, eq_b


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(
                    self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class Substitute(abc.ABC):
    def substitute(self, x: torch.Tensor):
        n_batch = x.size(0)
        n_x = x.size(-1)
        n_conv = len(self.blocks)

        x_out = list()
        x = x.permute(4, 0, 1, 2, 3).flatten(0, 1)
        for _, conv in self.blocks.items():
            x_out.append(conv(x))

        x_out = torch.cat(x_out, dim=0).unflatten(0, (n_x * n_conv, n_batch))

        if self.training:
            x_out = x_out[torch.randperm(x_out.size(0))]

        x_out = x_out.unflatten(0, (n_conv, n_x))
        return x_out.sum(1).permute(1, 2, 3, 4, 0)


# =================== Substitution Block

class SubConvBNBlock(nn.Module, Substitute):
    def __init__(self, in_channels, out_channels, kernel_size, n_block=3, stride=1, padding=0, bias=False, groups=1):
        super().__init__()
        self.conv_args = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias,
            'groups': groups,
        }
        self.conv_reparam = None
        self.n_block = n_block
        self.n_flow = 1

        self.blocks = nn.ModuleDict()
        for i in range(n_block):
            self.blocks.update({f'kxk_{i}': nn.Sequential(
                nn.Conv2d(**self.conv_args),
                nn.BatchNorm2d(out_channels),
            )})

    def forward(self, x: torch.Tensor):
        if self.conv_reparam:
            return self.conv_reparam(x)
        self.n_flow = x.size(-1)
        return self.substitute(x)

    def re_parameterization(self):
        self.conv_args['bias'] = True
        self.conv_reparam = nn.Conv2d(**self.conv_args)

        eq_k, eq_b = get_equivalent_kernel_bias(list(self.blocks.values()), self.n_flow)

        self.conv_reparam.weight.data = eq_k
        self.conv_reparam.bias.data = eq_b

        self.__delattr__('blocks')


class SubLinear(nn.Module):
    def __init__(self, in_features, out_features, n_block=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc_list = nn.ModuleList()
        self.fc = None
        self.n_block = n_block
        for _ in range(self.n_block):
            self.fc_list.append(nn.Linear(self.in_features, self.out_features))

    def forward(self, x):
        if self.fc:
            return self.fc(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            n_x = x.size(-1)
            b = x.size(0)
            n_fc = len(self.fc_list)

            x_out = list()

            x = rearrange(x, 'b c h w n -> (n b) h w c')
            for fc in self.fc_list:
                x_out.append(fc(x))

            x_out = torch.cat(x_out, dim=0).unflatten(0, (n_x * n_fc, b))

            if self.training:
                x_out = x_out[torch.randperm(x_out.size(0))]

            x_out = x_out.unflatten(0, (n_fc, n_x))

            return x_out.sum(1).permute(1, 4, 2, 3, 0)

    def re_parameterization(self):
        self.fc = nn.Linear(self.in_features, self.out_features)
        eq_k, eq_b = fuse_fc(self.fc_list, self.n_block)
        self.fc.weight.data = eq_k
        self.fc.bias.data = eq_b

        self.__delattr__('fc_list')


class SubClassifier(nn.Module):
    def __init__(self, in_features, out_features, n_block=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc_list = nn.ModuleList()
        self.fc = None
        self.n_block = n_block
        for _ in range(self.n_block):
            self.fc_list.append(nn.Linear(self.in_features, self.out_features))

    def forward(self, x):
        if self.fc:
            return self.fc(x)
        else:
            n_x = x.size(-1)
            b = x.size(0)
            n_fc = len(self.fc_list)

            x_out = list()

            x = rearrange(x, 'b c n -> (n b) c')
            for fc in self.fc_list:
                x_out.append(fc(x))

            x_out = torch.cat(x_out, dim=0)
            x_out = rearrange(x_out, '(n b) c -> n b c', n=n_x * n_fc, b=b)

            return x_out.sum(0)

    def re_parameterization(self):
        self.fc = nn.Linear(self.in_features, self.out_features)
        eq_k, eq_b = fuse_fc(self.fc_list, self.n_block)
        self.fc.weight.data = eq_k
        self.fc.bias.data = eq_b

        self.__delattr__('fc_list')


# ============================ Conv+MLP Layer

class SubConvMLP(nn.Module):
    def __init__(self, in_features=512, out_features=512, n_blocks=3, ratio=4.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features = int(in_features * ratio)
        self.n_blocks = n_blocks
        self.re_parameterized = False
        self.act = nn.ReLU(inplace=True)

        self.dw_conv1 = SubConvBNBlock(in_features, in_features, (3, 3), n_blocks, groups=in_features, padding=1)
        self.fc1 = SubLinear(in_features, hidden_features, n_blocks)
        self.dw_conv2 = SubConvBNBlock(hidden_features, hidden_features, (3, 3), n_blocks, groups=hidden_features,
                                       padding=1)
        self.fc2 = SubLinear(hidden_features, out_features, n_blocks)

    def forward(self, x):
        if self.re_parameterized:
            x = self.act(self.dw_conv1(x))
            x = self.act(self.fc1(x))
            x = self.act(self.dw_conv2(x))
            x = self.fc2(x)
            return x.sum((2, 3))
        else:
            xs = self.dw_conv1(x.unsqueeze(-1))
            xs = self.guided_activation(xs)
            xs = self.fc1(xs)
            xs = self.guided_activation(xs)
            xs = self.dw_conv2(xs)
            xs = self.guided_activation(xs)
            xs = self.fc2(xs)
            return xs.sum((2, 3))
            # xs = xs.sum(-1)
            # return self.pool(xs).flatten(1, 3)

    def guided_activation(self, xs):
        x = torch.sum(xs, dim=4).squeeze(-1)
        x = self.act(x)

        dead_idx = (x != 0).float()
        xs = torch.mul(xs, dead_idx.unsqueeze(-1))
        return xs


class SubConvMLP1D(nn.Module):
    def __init__(self, in_features=512, out_features=512, n_blocks=3, ratio=4.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features = int(in_features * ratio)
        self.n_blocks = n_blocks
        self.re_parameterized = False
        self.act = nn.ReLU(inplace=True)

        self.dw_conv1 = SubConvBNBlock(in_features, in_features, (1, 1), n_blocks)
        self.fc1 = SubLinear(in_features, hidden_features, n_blocks)
        self.dw_conv2 = SubConvBNBlock(hidden_features, hidden_features, (1, 1), n_blocks)
        self.fc2 = SubLinear(hidden_features, out_features, n_blocks)

    def forward(self, x):
        if self.re_parameterized:
            x = self.act(self.dw_conv1(x.unsqueeze(-1).unsqueeze(-1)))
            x = self.act(self.fc1(x))
            x = self.act(self.dw_conv2(x))
            x = self.fc2(x)
            return x.sum((2, 3))
        else:
            xs = self.dw_conv1(x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            xs = self.guided_activation(xs)
            xs = self.fc1(xs)
            xs = self.guided_activation(xs)
            xs = self.dw_conv2(xs)
            xs = self.guided_activation(xs)
            xs = self.fc2(xs)
            return xs.sum((2, 3))
            # xs = xs.sum(-1)
            # return self.pool(xs).flatten(1, 3)

    def guided_activation(self, xs):
        x = torch.sum(xs, dim=4)
        x = self.act(x)

        dead_idx = (x != 0).float()
        xs = torch.mul(xs, dead_idx.unsqueeze(-1))
        return xs


class SubMLP(nn.Module):
    def __init__(
            self,
            in_features=512,
            n_blocks=3,
            N=14,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.hidden_features1 = int(in_features * 2)
        self.hidden_features2 = int(in_features * 4)
        self.n_blocks = n_blocks
        self.N = N

        self.dw_conv1 = None
        self.dw_conv2 = None
        self.fc1 = None
        self.fc2 = None

        self.fc_list1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_features, self.hidden_features1, (1, 1), bias=False),
                nn.BatchNorm2d(self.hidden_features1),
            ) for _ in range(n_blocks)])
        self.fc_list2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features1, self.hidden_features2, (1, 1), bias=False),
                nn.BatchNorm2d(self.hidden_features2),
            ) for _ in range(n_blocks)])
        self.act = nn.ReLU()
        self.fc_list3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features2, self.hidden_features1, (1, 1), bias=False),
                nn.BatchNorm2d(self.hidden_features1),
            ) for _ in range(n_blocks)])
        self.fc_list4 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.hidden_features1, self.out_features, (1, 1), bias=False),
                nn.BatchNorm2d(self.out_features),
            ) for _ in range(n_blocks)])

    def forward(self, x):
        if self.fc1:
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            out = self.fc4(x)
            return out
        else:
            x = x.unsqueeze(-1).unsqueeze(-1)
            xs = self.substitute(x.unsqueeze(-1), self.fc_list1, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list2, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list3, self.training)
            xs = self.guided_activation(xs)
            xs = self.substitute(xs, self.fc_list4, self.training)
            out = xs
        return out.squeeze(-2).squeeze(-2)

    def substitute(self, x, fc_layer, training):
        n_batch = x.size(0)
        n_x = x.size(-1)
        n_conv = len(fc_layer)

        p_idx = 0
        x_out = [0] * n_conv
        if training:
            rand_idx = torch.randperm(n_conv * n_x) % n_conv
        else:
            rand_idx = torch.arange(n_conv * n_x) % n_conv

        x = x.permute(4, 0, 1, 2, 3).flatten(0, 1)
        for conv in fc_layer:
            _out = conv(x).unflatten(0, (n_x, n_batch))

            for i, idx in enumerate(rand_idx[p_idx * n_x: (p_idx + 1) * n_x]):
                x_out[idx] = x_out[idx] + _out[i]
            p_idx = p_idx + 1

        x_out = torch.stack(x_out, dim=0)
        return x_out.permute(1, 2, 3, 4, 0)

    def guided_activation(self, xs):
        x = torch.sum(xs, dim=4).squeeze(-1)
        x = self.act(x)

        dead_idx = x == 0
        xs[dead_idx] = 0
        return xs

    def fc_list_re_parameterization(self, in_features, out_features, fc_list, scale):
        _fc = nn.Linear(in_features, out_features)
        eq_k, eq_b = get_equivalent_kernel_bias(fc_list, scale)
        _fc.weight.data = eq_k.flatten(1, 3)
        _fc.bias.data = eq_b
        return _fc

    def re_parameterization(self):
        self.fc1 = self.fc_list_re_parameterization(self.in_features, self.hidden_features1, self.fc_list1, 1)
        self.fc2 = self.fc_list_re_parameterization(self.hidden_features1, self.hidden_features2, self.fc_list2,
                                                    self.n_blocks)
        self.fc3 = self.fc_list_re_parameterization(self.hidden_features2, self.hidden_features1, self.fc_list3,
                                                    self.n_blocks)
        self.fc4 = self.fc_list_re_parameterization(self.hidden_features1, self.out_features, self.fc_list4,
                                                    self.n_blocks)

        self.__delattr__('fc_list1')
        self.__delattr__('fc_list2')
        self.__delattr__('fc_list3')
        self.__delattr__('fc_list4')


def rand_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.running_mean, 0, 0.1)
            nn.init.uniform_(m.running_var, 0, 0.1)
            nn.init.uniform_(m.weight, 0, 0.1)
            nn.init.uniform_(m.bias, 0, 0.1)
    return model


if __name__ == '__main__':
    fc = SubMLP(10, 3)
    fc = rand_bn(fc)
    fc.eval()

    x = torch.rand(2, 10)
    prob = fc(x).sum(-1)
    from src.models.adapter.ns_adapter import deploy

    deploy(fc)
    reprob = fc(x)

    print(reprob.shape, prob.shape)

    print(((reprob - prob) ** 2).mean())
