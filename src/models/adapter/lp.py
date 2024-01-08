import torch

import src.models.clip as clip
from src.utils.registry import register_model


def forward(self, image, text=None):
    image_features = self.encode_image(image)
    image_features = self.classifier(image_features)

    return image_features


@register_model
def LP(backbone='ViT-B16', freeze=False, classifier=True, **kwargs):
    assert classifier
    model, _ = clip.load(backbone)
    if 'B16' in backbone or 'B32' in backbone:
        dim = 512
    else:
        dim = 768

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True

    if classifier:
        model.__setattr__('classifier', torch.nn.Linear(dim, 1000))
        forward_bound_method = forward.__get__(model, model.__class__)
        setattr(model, 'forward', forward_bound_method)

    return model


if __name__ == '__main__':
    model = LP(freeze=True, finetune=True)

    o = model(torch.rand(2, 3, 224, 224), torch.ones(2, 77, dtype=torch.long))
    print(len(o))
    print(o[0].shape)
