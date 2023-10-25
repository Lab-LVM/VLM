from src.utils.registry import register_feature_engine
from .modeling_clip import CLIPModel
from .tokenization_clip_fast import CLIPTokenizerFast


def init_vision_weights(model):
    from torch import nn

    model.vision_model.apply(model._init_weights)
    nn.init.normal_(
        model.visual_projection.weight,
        std=model.vision_embed_dim ** -0.5 * model.config.initializer_factor,
    )
    return model


def raise_not_implemented_finetune(model_name):
    raise NotImplementedError(f'{model_name} is not implemented finetune')


@register_feature_engine
def CLIP(pretrained=False, finetuned=False, freeze=False, **kwargs):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    if not pretrained:
        model = init_vision_weights(model)

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    if finetuned:
        import torch
        model.__setattr__('adapter', torch.nn.Linear(512, kwargs['num_classes']))

    model.set_loss_fn(kwargs.get('loss_fn', 'clip_loss'))
    return dict(model=model, tokenizer=tokenizer)


@register_feature_engine
def Tip(pretrained=False, finetuned=False, freeze=False, **kwargs):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    if not pretrained:
        model = init_vision_weights(model)

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    if finetuned:
        import torch
        model.__setattr__('adapter', torch.nn.Linear(512, kwargs['num_classes']))

    model.set_loss_fn(kwargs.get('loss_fn', 'clip_loss'))
    return dict(model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    model = CLIP(pretrained=False, loss_fn='sigmoid_loss')
    print(model['model'].loss_fn)
