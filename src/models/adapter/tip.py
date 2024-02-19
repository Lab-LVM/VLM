from src.models.openai_clip import load_clip_from_transformers
from src.utils.registry import register_model


@register_model
def Tip(backbone='ViT-B16', freeze=False, finetune=False, **kwargs):
    assert finetune
    model = load_clip_from_transformers(backbone)
    dim = model.projection_dim

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True

    if finetune:
        import torch
        model.__setattr__('adapter', torch.nn.Linear(dim, kwargs['num_classes'] * kwargs['n_shot']))

    return model
