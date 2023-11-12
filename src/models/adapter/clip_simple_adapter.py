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


def mlp():
    return torch.nn.Sequential(
        torch.nn.Linear(512, 512 * 4),
        torch.nn.GELU(),
        torch.nn.Linear(512 * 4, 512),
        torch.nn.LayerNorm(512),
    )


@register_model
def CLIP_SimpleAdapter(backbone='ViT-B32', freeze=False, finetune=False, language_adapter=False, vision_adapter=False,
                       classifier=False, **kwargs):
    assert finetune
    model, _ = clip.load(backbone)

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    if finetune:
        assert language_adapter or vision_adapter
        model.logit_scale.require_grad = True
        import torch

        if language_adapter:
            model.__setattr__('language_adapter', mlp())
            encode_text_bound_method = encode_text.__get__(model, model.__class__)
            setattr(model, 'encode_text', encode_text_bound_method)

        if vision_adapter:
            model.__setattr__('vision_adapter', mlp())
            encode_image_bound_method = encode_image.__get__(model, model.__class__)
            setattr(model, 'encode_image', encode_image_bound_method)

        if classifier:
            model.__setattr__('classifier', torch.nn.Linear(512, 1000))
            forward_bound_method = forward.__get__(model, model.__class__)
            setattr(model, 'forward', forward_bound_method)

    return model
