import torch
from einops.layers.torch import Rearrange

import src.models.clip as clip
from src.utils.registry import register_model


def ViT_feature_map_forward(self, x: torch.Tensor):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
        dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    token = self.ln_post(x[:, 0, :])

    if self.proj is not None:
        token = token @ self.proj

    return token, x[:, 1:, :]


def ViT_feature_map_encode_image(self, image):
    x, feature_map = self.visual(image.type(self.dtype))
    return self.vision_token_adapter(x) + self.vision_feature_adapter(feature_map) + x


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


def encode_image_train(self, x):
    return self.vision_adapter(x) + x


def encode_text_train(self, x):
    return self.language_adapter(x) + x


def forward(self, image, text):
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text, self.classifier(image_features)


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


def dw_conv(dim):
    return torch.nn.Sequential(
        torch.nn.Conv2d(dim, dim, (3, 3), padding=1, groups=dim, bias=False),
        torch.nn.BatchNorm2d(dim),
        torch.nn.GELU(),
    )


def pw_conv(dim, hidden):
    group = dim if hidden > dim else hidden
    return torch.nn.Sequential(
        torch.nn.Conv2d(dim, hidden, (1, 1), groups=group, bias=False),
        torch.nn.BatchNorm2d(hidden),
        torch.nn.GELU(),
    )


def conv(dim, out_dim):
    return torch.nn.Sequential(
        Rearrange('b (h w) c -> b c h w', h=14, w=14),
        dw_conv(dim),
        pw_conv(dim, int(dim * 4)),
        pw_conv(int(dim * 4), out_dim),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )


@register_model
def Our(backbone='ViT-B16', freeze=False, finetune=False, language_adapter=False, vision_adapter=False,
        classifier=False, **kwargs):
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
            if kwargs.get('forward_backbone', False):
                encode_text_bound_method = encode_text.__get__(model, model.__class__)
            else:
                encode_text_bound_method = encode_text_train.__get__(model, model.__class__)
            setattr(model, 'encode_text', encode_text_bound_method)

        if vision_adapter:
            model.__setattr__('vision_adapter', mlp(dim=dim))
            if kwargs.get('forward_backbone', False):
                encode_image_bound_method = encode_image.__get__(model, model.__class__)
            else:
                encode_image_bound_method = encode_image_train.__get__(model, model.__class__)

            setattr(model, 'encode_image', encode_image_bound_method)

        if classifier:
            model.__setattr__('classifier', torch.nn.Linear(dim, 1000))
            forward_bound_method = forward.__get__(model, model.__class__)
            setattr(model, 'forward', forward_bound_method)

    if kwargs.get('return_feature', False):
        forward_bound_method = forward_features.__get__(model, model.__class__)
        setattr(model, 'forward', forward_bound_method)

    return model


@register_model
def OurConv(backbone='ViT-B16', freeze=False, finetune=False, **kwargs):
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
        model.__setattr__('language_adapter', mlp(dim=dim))
        setattr(model, 'encode_text', encode_text.__get__(model, model.__class__))

        model.__setattr__('vision_token_adapter', mlp(dim=dim))
        model.__setattr__('vision_feature_adapter', conv(768, dim))

        setattr(model, 'encode_image', ViT_feature_map_encode_image.__get__(model, model.__class__))
        setattr(model.visual, 'forward', ViT_feature_map_forward.__get__(model.visual, model.visual.__class__))

        forward_bound_method = forward_features.__get__(model, model.__class__)
        setattr(model, 'forward', forward_bound_method)

    return model


if __name__ == '__main__':
    model = OurConv(freeze=True, finetune=True)

    o = model(torch.rand(2, 3, 224, 224), torch.ones(2, 77, dtype=torch.long))
    print(len(o))
    print(o[0].shape)
