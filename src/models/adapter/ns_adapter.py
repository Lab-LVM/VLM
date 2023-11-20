import torch
from torch import nn

import src.models.clip as clip
from src.misc.metadata import format_bytes
from src.models.adapter.ns_adapter_sub_block import SubConvMLP, SubClassifier, SubConv6MLP
from src.utils.registry import register_model


def transformer_forward(self, x: torch.Tensor):
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

    x = self.ln_post(x[:, 1:, :])

    if self.proj is not None:
        x = x @ self.proj

    return x


def encode_image2(self, image):
    x = self.visual(image.type(self.dtype))
    H = W = int(x.size(1) ** 0.5)
    x = x.permute(0, 2, 1).unflatten(-1, (H, W))
    return self.vision_adapter(x)


def encode_image(self, image):
    x = self.visual(image.type(self.dtype))
    return self.vision_adapter(x)


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


def mlp2():
    return torch.nn.Sequential(
        torch.nn.Linear(512, 512 * 2),
        torch.nn.LayerNorm(512 * 2),
        torch.nn.GELU(),
        torch.nn.Linear(512 * 2, 512 * 4),
        torch.nn.LayerNorm(512 * 4),
        torch.nn.GELU(),
        torch.nn.Linear(512 * 4, 512 * 2),
        torch.nn.LayerNorm(512 * 2),
        torch.nn.GELU(),
        torch.nn.Linear(512 * 2, 512),
        torch.nn.LayerNorm(512),
    )


def deploy(model):
    for name, module in model.named_modules():
        if hasattr(module, 're_parameterization'):
            module.re_parameterization()
        if hasattr(module, 're_parameterized'):
            module.re_parameterized = True


@register_model
def CLIP_NSAdapter2(backbone='ViT-B32', freeze=False, finetune=False, language_adapter=False, vision_adapter=False,
                    classifier=False, **kwargs):
    assert finetune
    model, _ = clip.load(backbone)

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    if finetune:
        assert language_adapter or vision_adapter
        import torch

        transformer_forward_bound = transformer_forward.__get__(model.visual, model.visual.__class__)
        setattr(model.visual, 'forward', transformer_forward_bound)

        model.__setattr__('vision_adapter', SubConv6MLP(512, 512, 3, ratio=4.0))
        encode_image2_bound_method = encode_image2.__get__(model, model.__class__)
        setattr(model, 'encode_image', encode_image2_bound_method)

        model.__setattr__('classifier', SubClassifier(512, 1000, 3))
        forward_bound_method = forward.__get__(model, model.__class__)
        setattr(model, 'forward', forward_bound_method)

    return model


@register_model
def CLIP_NSAdapter(backbone='ViT-B32', freeze=False, finetune=False, language_adapter=False, vision_adapter=False,
                   classifier=False, **kwargs):
    assert finetune
    model, _ = clip.load(backbone)

    delattr(model, 'transformer')
    delattr(model, 'text_projection')
    delattr(model, 'logit_scale')

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    if finetune:
        assert language_adapter or vision_adapter
        import torch

        transformer_forward_bound = transformer_forward.__get__(model.visual, model.visual.__class__)
        setattr(model.visual, 'forward', transformer_forward_bound)

        model.__setattr__('vision_adapter', SubConvMLP(512, 512, 3, ratio=4.0))
        encode_image2_bound_method = encode_image2.__get__(model, model.__class__)
        setattr(model, 'encode_image', encode_image2_bound_method)

        model.__setattr__('classifier', SubClassifier(512, 1000, 3))
        forward_bound_method = forward.__get__(model, model.__class__)
        setattr(model, 'forward', forward_bound_method)

    return model


@register_model
def CLIP_BaseAdapter(backbone='ViT-B32', freeze=False, finetune=False, language_adapter=False, vision_adapter=False,
                     classifier=False, **kwargs):
    assert finetune
    model, _ = clip.load(backbone)

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    if finetune:
        assert language_adapter or vision_adapter
        import torch

        model.__setattr__('vision_adapter', mlp())
        encode_image_bound_method = encode_image.__get__(model, model.__class__)
        setattr(model, 'encode_image', encode_image_bound_method)

        model.__setattr__('classifier', torch.nn.Linear(512, 1000))
        forward_bound_method = forward.__get__(model, model.__class__)
        setattr(model, 'forward', forward_bound_method)

    return model


def count_parameters(model):
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return format_bytes(param), param


if __name__ == '__main__':
    model = CLIP_NSAdapter(freeze=True, finetune=True, vision_adapter=True, language_adapter=False)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.running_mean, 0, 0.1)
            nn.init.uniform_(m.running_var, 0, 0.1)
            nn.init.uniform_(m.weight, 0, 0.1)
            nn.init.uniform_(m.bias, 0, 0.1)

    input = torch.rand(2, 3, 224, 224)
    prob = model.classifier(model.encode_image(input))
    param = count_parameters(model)

    deploy(model)

    reprob = model.classifier(model.encode_image(input))
    reparam = count_parameters(model)

    print("Diff", ((reprob - prob) ** 2).mean())

    print("Parameter")
    print(f"{reparam[0]} / {param[0]} = {reparam[1] / param[1]:.3f}")
