import src.models.clip as clip
from src.utils.registry import register_model


@register_model
def Tip(backbone='ViT-B32', freeze=False, finetune=False, **kwargs):
    assert finetune
    model, _ = clip.load(backbone)

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    if finetune:
        import torch
        model.__setattr__('adapter', torch.nn.Linear(512, kwargs['num_classes'] * kwargs['n_shot']))

    return model
