import torch

import src.models.clip as clip
from src.utils.registry import register_model


def encode_image(self, image):
    x = self.visual(image.type(self.dtype))
    return self.vision_adapter(x) + x * self.alpha


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
    return self.language_adapter(x) + x * self.alpha


def encode_image_train(self, x):
    return self.vision_adapter(x) + x * self.alpha


def encode_text_train(self, x):
    return self.language_adapter(x) + x * self.alpha


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


def forward_features_prob(self, image, text):
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return image_features, text_features, self.classifier(image_features)


def mlp(dim=512):
    return torch.nn.Sequential(
        torch.nn.Linear(dim, dim * 4),
        torch.nn.GELU(),
        torch.nn.Linear(dim * 4, dim),
        torch.nn.LayerNorm(dim),
    )


@register_model
def Our(backbone='ViT-B16', freeze=False, finetune=False, language_adapter=False, vision_adapter=False,
        classifier=False, **kwargs):
    assert finetune
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
        # assert language_adapter or vision_adapter

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

        if kwargs.get('alpha', False):
            model.__setattr__('alpha', torch.nn.Parameter(torch.rand(1)))
        else:
            model.__setattr__('alpha', 1)

        if kwargs.get('return_feature', False):
            if classifier:
                forward_bound_method = forward_features_prob.__get__(model, model.__class__)
                setattr(model, 'forward', forward_bound_method)
            else:
                forward_bound_method = forward_features.__get__(model, model.__class__)
            setattr(model, 'forward', forward_bound_method)

    return model


if __name__ == '__main__':
    model = Our(eval=True, alpha=False, finetune=True, freeze=True, language_adapter=True, vision_adapter=True)

    o = model(torch.rand(2, 3, 224, 224), torch.ones(2, 77, dtype=torch.long))
    print(o.shape)
