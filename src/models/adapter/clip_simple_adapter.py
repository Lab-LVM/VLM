import torch

import src.models.clip as clip
from src.utils.registry import register_model


def encode_image(self, image):
    return self.vision_adapter(self.visual(image.type(self.dtype)))


def encode_text(self, text):
    x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    return self.language_adapter(x)


def large_linear():
    return torch.nn.Sequential(
        torch.nn.Linear(512, 512 * 4),
        torch.nn.GELU(),
        torch.nn.Linear(512 * 4, 512),
        torch.nn.LayerNorm(512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 512 * 4),
        torch.nn.GELU(),
        torch.nn.Linear(512 * 4, 512),
        torch.nn.LayerNorm(512),
    )


def base_linear():
    return torch.nn.Sequential(
        torch.nn.LayerNorm(512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 512 * 4),
        torch.nn.GELU(),
        torch.nn.Linear(512 * 4, 512),
        torch.nn.LayerNorm(512),
    )


def base_linear_nln():
    return torch.nn.Sequential(
        torch.nn.LayerNorm(512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 512 * 4),
        torch.nn.GELU(),
        torch.nn.Linear(512 * 4, 512),
    )


def linear():
    return torch.nn.Linear(512, 512)


@register_model
def CLIP_SimpleAdapter(backbone='ViT-B32', freeze=False, finetune=False, language_adapter=False, vision_adapter=False,
                       **kwargs):
    assert finetune
    model, _ = clip.load(backbone)

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    if finetune:
        assert language_adapter or vision_adapter
        model.logit_scale.require_grad = True
        import torch

        if kwargs.get('scale', None):
            if kwargs['scale'].lower() == 'base':
                linear_fn = base_linear
            if kwargs['scale'].lower() == 'base_nln':
                linear_fn = base_linear_nln
            elif kwargs['scale'].lower() == 'large':
                linear_fn = large_linear
        else:
            linear_fn = linear

        if language_adapter:
            model.__setattr__('language_adapter', linear_fn())
            encode_text_bound_method = encode_text.__get__(model, model.__class__)
            setattr(model, 'encode_text', encode_text_bound_method)

        if vision_adapter:
            model.__setattr__('vision_adapter', linear_fn())
            encode_image_bound_method = encode_image.__get__(model, model.__class__)
            setattr(model, 'encode_image', encode_image_bound_method)

    return model
