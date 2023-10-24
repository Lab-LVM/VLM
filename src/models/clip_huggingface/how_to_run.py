import torch
from transformers import CLIPConfig

from src.models.clip_huggingface.modeling_clip import CLIPModel
from src.models.clip_huggingface.tokenization_clip_fast import CLIPTokenizerFast


def normalize(features):
    return features / features.norm(p=2, dim=-1, keepdim=True)


config = CLIPConfig()
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel(config)

text_input = tokenizer(["a photo of a cat", "a image of a dog"], padding='max_length', truncation=True, return_tensors="pt")
text_feature = model.get_text_features(**text_input)
print('Tokens: ', text_input)
print('Inputs:', text_input.keys())
print('Text features:', text_feature.shape)

image_input = dict(pixel_values=torch.rand(2, 3, 224, 224))
image_feature = model.get_image_features(**image_input)
print('Inputs:', image_input.keys())
print('Vision features:', image_feature.shape)

text_feature = normalize(text_feature)
image_feature = normalize(image_feature)

logits_per_text = torch.matmul(text_feature, image_feature.t())
logits_per_image = logits_per_text.t()
