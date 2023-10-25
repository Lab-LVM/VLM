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


@register_model
def CLIP_SimpleAdapter(freeze=False, finetune=False, **kwargs):
    model, _ = clip.load("ViT-B/32")

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    if finetune:
        model.logit_scale.require_grad = True
        import torch
        model.__setattr__('language_adapter', torch.nn.Linear(512, 512))
        model.__setattr__('vision_adapter', torch.nn.Linear(512, 512))

        encode_text_bound_method = encode_text.__get__(model, model.__class__)
        encode_image_bound_method = encode_image.__get__(model, model.__class__)

        setattr(model, 'encode_text', encode_text_bound_method)
        setattr(model, 'encode_image', encode_image_bound_method)

    return model
