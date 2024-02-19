from typing import Optional

import torch
from transformers import CLIPTokenizerFast, CLIPModel

from src.utils.registry import register_model


def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
):
    image_embeds = self.encode_image(
        pixel_values=pixel_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    text_embeds = self.encode_text(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    # normalized features
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds, text_embeds


def encode_text(self, input_ids, attention_mask=None, position_ids=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
    # outputs = (last_hidden_state, pooled_output)
    text_outputs = self.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    return self.text_projection(text_outputs[1])


def encode_image(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None):
    # outputs = (last_hidden_state, pooled_output)
    vision_outputs = self.vision_model(
        pixel_values=pixel_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    return self.visual_projection(vision_outputs[1])


def load_clip_from_transformers(backbone):
    _MODELS = {
        'ViT-B32': 'clip-vit-base-patch32',
        'ViT-B16': 'clip-vit-base-patch16',
        'ViT-L14': 'clip-vit-large-patch14',
        'ViT-L14@336px': 'clip-vit-large-patch14-336',
    }
    model = CLIPModel.from_pretrained(f'openai/{_MODELS[backbone]}')

    forward_bound = forward.__get__(model, model.__class__)
    setattr(model, 'forward', forward_bound)

    encode_text_bound = encode_text.__get__(model, model.__class__)
    setattr(model, 'encode_text', encode_text_bound)

    encode_image_bound = encode_image.__get__(model, model.__class__)
    setattr(model, 'encode_image', encode_image_bound)

    return model


@register_model
def CLIP(backbone='ViT-B16', **kwargs):
    model = load_clip_from_transformers(backbone)

    if kwargs.get('freeze', False):
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True
    return model


@register_model
def CLIP_tokenizer(**kwargs):
    return CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')
