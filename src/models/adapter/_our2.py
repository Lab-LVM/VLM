import torch
from torch import nn

import src.models.clip as clip
from src.utils.registry import register_model


def encode_image(self, image):
    x = self.visual(image.type(self.dtype))
    return self.vision_adapter(x)


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


def encode_image_train(self, x):
    return self.vision_adapter(x)


def encode_text_train(self, x):
    return self.language_adapter(x)


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


class MLP(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
            torch.nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.layer(x) + x


class MLPNorm(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
            torch.nn.LayerNorm(dim),
        )
        self.ln = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.layer(x) + self.ln(x)


class MLP2(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 2, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 2, dim),
            torch.nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.layer(x) + x


class MLP3(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(1, 1, batch_first=True)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        B, C = x.size()
        qkv = self.qkv(x).reshape(B, 1, -1, 3)
        q, k, v = qkv.unbind(3)
        attn, _ = self.attn(q.mT, k.mT, v.mT, need_weights=False)

        return attn.squeeze(-1)


class MLP3P(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(1, 1, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        B, C = x.size()
        qkv = self.qkv(x).reshape(B, 1, -1, 3)
        q, k, v = qkv.unbind(3)
        attn, _ = self.attn(q.mT, k.mT, v.mT, need_weights=False)
        attn = self.proj(attn.squeeze(-1))

        return attn


class MLP3PA(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(1, 1, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        B, C = x.size()
        qkv = self.qkv(x).reshape(B, 1, -1, 3)
        q, k, v = qkv.unbind(3)
        attn, _ = self.attn(q.mT, k.mT, v.mT, need_weights=False)
        attn = self.proj(attn.squeeze(-1))

        return attn + x


class MLP3Attn(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(1, 1, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
            torch.nn.LayerNorm(dim),
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C = x.size()
        qkv = self.qkv(x).reshape(B, 1, -1, 3)
        q, k, v = qkv.unbind(3)
        attn, _ = self.attn(q.mT, k.mT, v.mT, need_weights=False)
        attn = self.proj(attn.squeeze(-1))
        x = x + attn
        x = x + self.ffn(x)
        return x


class MLP3AttnAdd(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(1, 1, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
            torch.nn.LayerNorm(dim),
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C = x.size()
        qkv = self.qkv(self.ln1(x)).reshape(B, 1, -1, 3)
        q, k, v = qkv.unbind(3)
        attn, _ = self.attn(q.mT, k.mT, v.mT, need_weights=False)
        attn = self.proj(attn.squeeze(-1))
        attn = self.ffn(self.ln2(attn))
        return attn + x


class MLP3A(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(1, 1, batch_first=True)
        self.qkv = nn.Linear(dim, 512 * 3, bias=False)

    def forward(self, x):
        B, C = x.size()
        qkv = self.qkv(x).reshape(B, 1, -1, 3)
        q, k, v = qkv.unbind(3)
        attn, _ = self.attn(q.mT, k.mT, v.mT, need_weights=False)

        return attn.squeeze(-1) + x


@register_model
def Our2(backbone='ViT-B16', freeze=False, finetune=False, language_adapter=False, vision_adapter=False, adapter=None,
         **kwargs):
    model, _ = clip.load(backbone)
    adapter_fn = None

    if adapter == 'MLP':
        adapter_fn = MLP
    elif adapter == 'MLP2':
        adapter_fn = MLP2
    elif adapter == 'MLP3':
        adapter_fn = MLP3
    elif adapter == 'MLP3A':
        adapter_fn = MLP3A
    elif adapter == 'MLP3P':
        adapter_fn = MLP3P
    elif adapter == 'MLP3PA':
        adapter_fn = MLP3PA
    elif adapter == 'MLP3Attn':
        adapter_fn = MLP3Attn
    elif adapter == 'MLP3AttnAdd':
        adapter_fn = MLP3AttnAdd
    elif adapter == 'MLPNorm':
        adapter_fn = MLPNorm

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
            model.__setattr__('language_adapter', adapter_fn(dim=dim))
            if kwargs.get('forward_backbone', False):
                encode_text_bound_method = encode_text.__get__(model, model.__class__)
            else:
                encode_text_bound_method = encode_text_train.__get__(model, model.__class__)
            setattr(model, 'encode_text', encode_text_bound_method)

        if vision_adapter:
            model.__setattr__('vision_adapter', adapter_fn(dim=dim))
            if kwargs.get('forward_backbone', False):
                encode_image_bound_method = encode_image.__get__(model, model.__class__)
            else:
                encode_image_bound_method = encode_image_train.__get__(model, model.__class__)

            setattr(model, 'encode_image', encode_image_bound_method)

        if kwargs.get('return_feature', False):
            forward_bound_method = forward_features.__get__(model, model.__class__)
            setattr(model, 'forward', forward_bound_method)

    return model


if __name__ == '__main__':
    model = Our2(finetune=True, freeze=True, language_adapter=True, vision_adapter=True, return_feature=True,
                 adapter='MLP2')

    im, te = torch.rand(2, 512), torch.rand(2, 512)
    o = model(im, te)
    print(len(o))
    print(o[0].shape)
