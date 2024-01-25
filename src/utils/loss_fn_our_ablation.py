import torch
from torch import nn

from src.utils.loss_function import gather_all_features


class AugmentedContrastiveLossAblation(nn.Module):
    def __init__(self, fn_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = 0
        self.world_size = 1
        self.set_forward(fn_name)

    def set_forward(self, fn_name):
        self._forward = self.__getattribute__('_' + fn_name)

    def SoftCE(self, logits, targets):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        if logits.size(0) > logits.size(1):
            targets = targets[:, :logits.size(1)]
        elif logits.size(0) < logits.size(1):
            targets = targets[:logits.size(0), :]

        cardinality = torch.sum(targets, dim=1)
        loss = torch.sum(-targets * torch.nn.functional.log_softmax(logits, dim=-1), dim=-1) / cardinality
        return loss.mean()

    def NTX(self, logits, targets):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        if logits.size(0) > logits.size(1):
            targets = targets[:, :logits.size(1)]
        elif logits.size(0) < logits.size(1):
            targets = targets[:logits.size(0), :]

        cardinality = torch.sum(targets, dim=1)
        logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0]) + 1e-5
        loss = torch.sum(-targets * torch.nn.functional.log_softmax(logits / cardinality, dim=-1), dim=-1)
        return loss.mean()

    def ACL(self, logits, targets):
        mask_same_class = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        if logits.size(0) > logits.size(1):
            mask_same_class = mask_same_class[:, :logits.size(1)]
        elif logits.size(0) < logits.size(1):
            mask_same_class = mask_same_class[:logits.size(0), :]

        mask_inverse = 1 - mask_same_class
        cardinality = torch.sum(mask_same_class, dim=1)

        exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0]) + 1e-5
        log_prob = -torch.log(exp_logits / torch.sum(exp_logits * mask_inverse, dim=1, keepdim=True))
        sample_wise_loss = torch.sum(log_prob * mask_same_class, dim=1) / cardinality

        return torch.mean(sample_wise_loss)

    @staticmethod
    def generate_logits(image_feature, text_feature, logit_scale):
        logits_per_image = logit_scale * torch.mm(image_feature, text_feature.t())

        half = image_feature.size(0) // 2
        logits_image_self = logit_scale * torch.mm(image_feature[:half], image_feature[half:].t())
        logits_text_self = logit_scale * torch.mm(text_feature[:half], text_feature[half:].t())

        return logits_per_image, logits_image_self, logits_text_self

    @staticmethod
    def split_to_quadrant(tensor):
        """
        Tensor must be logits_per_image
        """
        size = tensor.size(0) // 2
        split = torch.split(tensor, size, dim=0)
        IT, It = torch.split(split[0], size, dim=1)
        iT, it = torch.split(split[1], size, dim=1)
        return IT, It, iT, it

    def forward(self, image_features, text_features, targets, logit_scale):
        if self.world_size > 1:
            image_features, text_features, targets = gather_all_features(image_features, text_features, targets)

        logits_per_image, logits_image_self, logits_text_self = self.generate_logits(image_features, text_features,
                                                                                     logit_scale)
        self.logits_per_image = logits_per_image
        self.targets = targets

        self_targets = torch.arange(logits_image_self.size(0), device=logits_image_self.device, dtype=torch.long)

        targets = torch.split(targets, targets.size(0) // 2, dim=0)[0]
        IT, It, iT, it = self.split_to_quadrant(logits_per_image)

        return self._forward(IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self)

    def _ntype1(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        return (self.NTX(IT, targets) + self.NTX(IT.t(), targets)) / 2

    def _ntype2(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        return (self.NTX(it, targets) + self.NTX(it.t(), targets)) / 2

    def _ntype3(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        return (self.NTX(it, targets) + self.NTX(it.t(), targets) + self.NTX(IT, targets) + self.NTX(IT.t(), targets)) / 4

    def _ctype1(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        return (self.SoftCE(IT, targets) + self.SoftCE(IT.t(), targets)) / 2

    def _ctype2(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        return (self.SoftCE(it, targets) + self.SoftCE(it.t(), targets)) / 2

    def _ctype3(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        return (self.SoftCE(it, targets) + self.SoftCE(it.t(), targets) + self.SoftCE(IT, targets) + self.SoftCE(IT.t(), targets)) / 4

    def _ctype4(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        return (self.SoftCE(iT, targets) + self.SoftCE(iT.t(), targets)) / 2

    def _ctype5(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        targets = (targets.unsqueeze(-1) == targets.unsqueeze(0)).float()
        return (self.SoftCE(It, targets) + self.SoftCE(It.t(), targets)) / 2

    def _ctype6(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        cat = torch.cat([IT, It])
        cat_target = torch.cat([targets, targets])
        return (self.SoftCE(cat, cat_target) + self.SoftCE(cat.t(), cat_target)) / 2

    def _ctype7(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        cat = torch.cat([IT, It])
        cat_target = torch.cat([targets, targets])
        return (self.SoftCE(cat, cat_target) + self.SoftCE(cat.t(), cat_target)) / 2

    def _atype1(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        return (self.ACL(IT, targets) + self.ACL(IT.t(), targets)) / 2

    def _atype2(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        return (self.ACL(it, targets) + self.ACL(it.t(), targets)) / 2

    def _atype3(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        return (self.ACL(it, targets) + self.ACL(it.t(), targets) + self.ACL(IT, targets) + self.ACL(IT.t(),
                                                                                                     targets)) / 4

    def _atype4(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        return (self.ACL(iT, targets) + self.ACL(iT.t(), targets)) / 2

    def _atype5(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        return (self.ACL(It, targets) + self.ACL(It.t(), targets)) / 2

    def _atype6(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        cat = torch.cat([IT, iT])
        cat_target = torch.cat([targets, targets])
        return (self.ACL(cat, cat_target) + self.ACL(cat.t(), cat_target)) / 2

    def _atype7(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        cat = torch.cat([IT, It])
        cat_target = torch.cat([targets, targets])
        return (self.ACL(cat, cat_target) + self.ACL(cat.t(), cat_target)) / 2

    def _aatype1(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        return (self._atype3(IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self) * 4 +
                self.ACL(iT, targets) + self.ACL(iT.t(), targets)) / 6

    def _aatype2(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        return (self._atype3(IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self) * 4 +
                self.ACL(It, targets) + self.ACL(It.t(), targets)) / 6

    def _aatype3(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        x = self.logits_per_image
        y = self.targets
        return (self.ACL(x, y) + self.ACL(x.t(), y)) / 2

    def _aastype1(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        x = self.logits_per_image
        y = self.targets
        return (self.ACL(x, y) + self.ACL(x.t(), y)) / 2

    def _aastype2(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        x = self.logits_per_image
        y = self.targets
        return (self.ACL(x, y) + self.ACL(x.t(), y) + self.ACL(logits_image_self, self_targets)) / 3

    def _aastype3(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        x = self.logits_per_image
        y = self.targets
        return (self.ACL(x, y) + self.ACL(x.t(), y) + self.ACL(logits_text_self, self_targets)) / 3

    def _aastype4(self, IT, It, iT, it, targets, self_targets, logits_image_self, logits_text_self):
        # catI = torch.cat([IT, It], dim=1)
        # cati = torch.cat([iT, it], dim=1)
        # x = torch.cat([catI, cati], dim=0)
        # y = torch.cat([targets, targets], dim=0)
        x = self.logits_per_image
        y = self.targets
        return (self.ACL(x, y) + self.ACL(x.t(), y) +
                self.ACL(logits_image_self, self_targets) + self.ACL(logits_text_self, self_targets)) / 4


if __name__ == '__main__':
    fn = AugmentedContrastiveLossAblation('_aastype1')

    img_f = torch.rand(8, 10)
    text_f = torch.rand(8, 10)
    targets = torch.tensor([0, 1, 2, 1, 0, 1, 2, 1], dtype=torch.long)

    loss1 = fn(img_f, text_f, targets, 100.)

    fn.set_forward('_aatype3')
    loss2 = fn(img_f, text_f, targets, 100.)

    print(loss1, loss2)

    # fn.set_forward('_atype3')
    # loss3 = fn(img_f, text_f, targets, 100.)
    #
    # print(loss1, loss2, loss3, (loss1 + loss2) / 2)
#
