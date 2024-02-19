import torch

from src.models.openai_clip import load_clip_from_transformers
from src.utils.registry import register_model


def forward(self, image, text=None):
    image_features = self.encode_image(image)
    image_features = self.classifier(image_features)
    return image_features


@register_model
def LP(backbone='ViT-B16', freeze=False, num_classes=1000, **kwargs):
    model = load_clip_from_transformers(backbone)
    dim = model.projection_dim

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True

    model.__setattr__('classifier', torch.nn.Linear(dim, num_classes))
    forward_bound_method = forward.__get__(model, model.__class__)
    setattr(model, 'forward', forward_bound_method)

    return model


if __name__ == '__main__':
    model = LP()

    o = model(torch.rand(2, 3, 224, 224), torch.ones(2, 77, dtype=torch.long))
    # print(len(o))
    # print(o[0].shape)
