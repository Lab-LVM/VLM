# https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py

import torch.distributed.nn
import torch.nn as nn
from torch import distributed as dist
from torch.nn import functional as F


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
):
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class CLIPLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, y, logit_scale):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

    def forward_by_logits(self, logits_per_image, logits_per_text):
        device = logits_per_image.device
        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2


class CoCaLoss(CLIPLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=-100,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale):
        clip_loss = torch.tensor(0)

        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        return clip_loss, caption_loss


class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)


def all_gather(tensor):
    return AllGatherFunction.apply(tensor)


def gather_all_features(*features):
    return [all_gather(feature) for feature in features]


def soft_cross_entropy(logit, target):
    loss = torch.sum(-target * F.log_softmax(logit, dim=-1), dim=-1)
    return loss.mean()


class SoftContrastiveLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = 0
        self.world_size = 1

    def SoftCE(self, logits, targets):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        if logits.size(0) > logits.size(1):
            targets = targets[:, :logits.size(1)]
        elif logits.size(0) < logits.size(1):
            targets = targets[:logits.size(0), :]

        cardinality = torch.sum(targets, dim=1)
        loss = torch.sum(-targets * torch.nn.functional.log_softmax(logits, dim=-1), dim=-1) / cardinality
        return loss.mean()

    def generate_logits(self, image_feature, text_feature, logit_scale):
        logits_per_image = logit_scale * torch.mm(image_feature, text_feature.t())
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, targets, logit_scale):
        if self.world_size > 1:
            image_features, text_features, targets = gather_all_features(image_features, text_features, targets)

        logits_per_image, logits_per_text = self.generate_logits(image_features, text_features, logit_scale)

        loss = (self.SoftCE(logits_per_image, targets) + self.SoftCE(logits_per_text, targets)) / 2
        return loss


class BCELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = 0
        self.world_size = 1
        self.BCE = nn.BCEWithLogitsLoss()

    def generate_logits(self, image_feature, text_feature, logit_scale):
        logits_per_image = torch.mm(image_feature, text_feature.t())
        logits_per_text = logits_per_image.t()

        half = image_feature.size(0) // 2
        logits_image_self = torch.mm(image_feature[:half], image_feature[half:].t())

        return logits_per_image, logits_per_text, logits_image_self

    def forward(self, image_features, text_features, targets, logit_scale):
        if self.world_size > 1:
            image_features, text_features, targets = gather_all_features(image_features, text_features, targets)

        logits_per_image, logits_per_text, logits_image_self = self.generate_logits(image_features, text_features,
                                                                                    logit_scale)

        self_targets = torch.arange(logits_image_self.size(0), device=logits_image_self.device, dtype=torch.long)

        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        self_targets = (self_targets.unsqueeze(-1) == self_targets.unsqueeze(0)).float()

        loss = (self.BCE(logits_per_image, targets) + self.BCE(logits_per_text, targets)
                + self.BCE(logits_image_self, self_targets)) / 3
        return loss


class AugCL2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = 0
        self.world_size = 1

    def ACL2(self, logits, targets):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        mask_inverse = 1 - targets
        cardinality = torch.sum(targets, dim=1)

        prob = torch.nn.functional.softmax(logits, dim=-1)

        exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0]) + 1e-5
        calibration = (torch.sum(exp_logits * targets, dim=1, keepdim=True) /
                       torch.sum(exp_logits * mask_inverse, dim=1, keepdim=True))

        log_prob = torch.log(prob * calibration)

        loss = torch.sum(-targets * log_prob, dim=-1) / cardinality
        return loss.mean()

    def generate_logits(self, image_feature, text_feature, logit_scale, targets):
        logits_per_image = logit_scale * torch.mm(image_feature, text_feature.t())
        logits_per_text = logits_per_image.t()

        half = image_feature.size(0) // 2
        logits_image_self = logit_scale * torch.mm(image_feature[:half], image_feature[half:].t())
        half_target = targets[:half]

        return logits_per_image, logits_per_text, logits_image_self, half_target

    def forward(self, image_features, text_features, targets, logit_scale):
        if self.world_size > 1:
            image_features, text_features, targets = gather_all_features(image_features, text_features, targets)

        logits_per_image, logits_per_text, logits_image_self, half_target = self.generate_logits(image_features,
                                                                                                 text_features,
                                                                                                 logit_scale,
                                                                                                 targets,
                                                                                                 )

        loss = (self.ACL2(logits_per_image, targets) + self.ACL2(logits_per_text, targets)
                + self.ACL2(logits_image_self, half_target)) / 3
        return loss


class IndomainOutdomainContrastiveLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = 0
        self.world_size = 1

    def SCL(self, logits, targets):
        mask_same_class = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        mask_inverse = 1 - mask_same_class

        cardinality = torch.sum(mask_same_class, dim=1)

        exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0]) + 1e-5
        log_prob = -torch.log(exp_logits / torch.sum(exp_logits * mask_inverse, dim=1, keepdim=True))
        sample_wise_loss = torch.sum(log_prob * mask_same_class, dim=1) / cardinality

        return torch.mean(sample_wise_loss)

    def generate_logits(self, image_feature, text_feature, logit_scale):
        logits_per_image = logit_scale * torch.mm(image_feature, text_feature.t())
        logits_per_text = logits_per_image.t()

        half = image_feature.size(0) // 2
        logits_image_self = logit_scale * torch.mm(image_feature[:half], image_feature[half:].t())

        return logits_per_image, logits_per_text, logits_image_self

    def forward(self, image_features, text_features, targets, logit_scale):
        if self.world_size > 1:
            image_features, text_features, targets = gather_all_features(image_features, text_features, targets)

        logits_per_image, logits_per_text, logits_image_self = self.generate_logits(image_features, text_features,
                                                                                    logit_scale)

        self_targets = torch.arange(logits_image_self.size(0), device=logits_image_self.device, dtype=torch.long)

        loss = (self.SCL(logits_per_image, targets) + self.SCL(logits_per_text, targets)
                + self.SCL(logits_image_self, self_targets)) / 3
        return loss


class IndomainOutdomainContrastiveLoss2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = 0
        self.world_size = 1

    def SCL(self, logits, targets, half=False):
        mask_same_class = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        mask_inverse = 1 - mask_same_class
        if half:
            mask_inverse = mask_inverse + mask_same_class * 0.5
            mask_same_class = mask_same_class * 0.5
        cardinality = torch.sum(mask_same_class, dim=1)

        log_prob = -torch.nn.functional.log_softmax(logits, dim=-1)
        sample_wise_loss = torch.sum(log_prob * mask_same_class, dim=1) / cardinality

        return torch.mean(sample_wise_loss)

    def generate_logits(self, image_feature, text_feature, logit_scale):
        logits_per_image = logit_scale * torch.mm(image_feature, text_feature.t())
        logits_per_text = logits_per_image.t()

        half = image_feature.size(0) // 2
        logits_image_self = logit_scale * torch.mm(image_feature[:half], image_feature[half:].t())

        return logits_per_image, logits_per_text, logits_image_self

    def forward(self, image_features, text_features, targets, logit_scale):
        if self.world_size > 1:
            image_features, text_features, targets = gather_all_features(image_features, text_features, targets)

        logits_per_image, logits_per_text, logits_image_self = self.generate_logits(image_features, text_features,
                                                                                    logit_scale)
        half = targets.size(0) // 2
        self_targets = targets[:half]

        loss = (self.SCL(logits_per_image, targets) + self.SCL(logits_per_text, targets)
                + self.SCL(logits_image_self, self_targets)) / 3
        return loss


class SupervisedContrastiveLossMultiProcessing(nn.Module):
    def __init__(self):
        super(SupervisedContrastiveLossMultiProcessing, self).__init__()
        self.rank = 0
        self.world_size = 1

    def SCL(self, dot_product_tempered, targets):
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_same_class = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        mask_anchor_out = 1 - mask_same_class

        cardinality_per_samples = torch.sum(mask_same_class, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_same_class, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        return supervised_contrastive_loss

    def forward(self, image_features, text_features, targets, logit_scale):
        if self.world_size > 1:
            image_features, text_features, targets = gather_all_features(image_features, text_features, targets)

        logits_per_image = logit_scale * torch.mm(image_features, text_features.t())
        logits_per_text = logits_per_image.t()

        return self.SCL(logits_per_image, targets) + self.SCL(logits_per_text, targets)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self):
        super(SupervisedContrastiveLoss, self).__init__()
        self.rank = 0
        self.world_size = 1

    def forward(self, logits, targets):
        dot_product_tempered = logits
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_same_class = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        mask_anchor_out = 1 - mask_same_class

        cardinality_per_samples = torch.sum(mask_same_class, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_same_class, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        return supervised_contrastive_loss


class OriginalSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(OriginalSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss


if __name__ == '__main__':
    import torch

    prob = torch.rand(6, 6)
    target = torch.arange(0, 6)
    target[1] += 2
    loss_fn = AugCL2()

    loss_fn(prob, prob, target, 100)
