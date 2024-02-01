import torch

import src.models.clip as clip
from src.utils.registry import register_model


def encode_image(self, image):
    x = self.visual(image.type(self.dtype))
    return self.vision_adapter(x) + x


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
    return self.language_adapter(x) + x


def forward_features(self, image, text):
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return image_features, text_features


def mlp(dim=512):
    return torch.nn.Sequential(
        torch.nn.Linear(dim, dim * 4),
        torch.nn.GELU(),
        torch.nn.Linear(dim * 4, dim),
        torch.nn.LayerNorm(dim),
    )


@register_model
def Our2(backbone='ViT-B16', freeze=False, language_adapter=False, vision_adapter=False, return_feature=True,
         finetune=False, **kwargs):
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

    if finetune:
        if language_adapter:
            model.__setattr__('language_adapter', mlp(dim=dim))
            encode_text_bound_method = encode_text.__get__(model, model.__class__)
            setattr(model, 'encode_text', encode_text_bound_method)

        if vision_adapter:
            model.__setattr__('vision_adapter', mlp(dim=dim))
            encode_image_bound_method = encode_image.__get__(model, model.__class__)
            setattr(model, 'encode_image', encode_image_bound_method)

    if return_feature:
        forward_bound_method = forward_features.__get__(model, model.__class__)
        setattr(model, 'forward', forward_bound_method)

    return model


if __name__ == '__main__':
    model = Our2(freeze=True, finetune=True)

    o = model(torch.rand(2, 3, 224, 224), torch.ones(2, 77, dtype=torch.long))
    print(len(o))
    print(o[0].shape)
