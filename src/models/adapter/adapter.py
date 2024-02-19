import torch

from src.models.openai_clip import load_clip_from_transformers
from src.utils.registry import register_model


def encode_text(self, input_ids, attention_mask=None, position_ids=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
    text_outputs = self.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    text_output = self.text_projection(text_outputs[1])
    return self.language_adapter(text_output) + text_output


def encode_image(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None):
    vision_outputs = self.vision_model(
        pixel_values=pixel_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    vision_output = self.visual_projection(vision_outputs[1])
    return self.vision_adapter(vision_output) + vision_output


def mlp(dim=512):
    return torch.nn.Sequential(
        torch.nn.Linear(dim, dim * 4),
        torch.nn.GELU(),
        torch.nn.Linear(dim * 4, dim),
        torch.nn.LayerNorm(dim),
    )


@register_model
def Our2(backbone='ViT-B16', freeze=False, language_adapter=False, vision_adapter=False, **kwargs):
    model = load_clip_from_transformers(backbone)
    dim = model.projection_dim

    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True

    if language_adapter:
        model.__setattr__('language_adapter', mlp(dim=dim))
        encode_text_bound_method = encode_text.__get__(model, model.__class__)
        setattr(model, 'encode_text', encode_text_bound_method)

    if vision_adapter:
        model.__setattr__('vision_adapter', mlp(dim=dim))
        encode_image_bound_method = encode_image.__get__(model, model.__class__)
        setattr(model, 'encode_image', encode_image_bound_method)

    return model


if __name__ == '__main__':
    model = Our2(freeze=True, finetune=True)

    o = model(torch.rand(2, 3, 224, 224), torch.ones(2, 77, dtype=torch.long))
    print(len(o))
    print(o[0].shape)
