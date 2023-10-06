import torch
from transformers import CLIPConfig

from src.models.clip.modeling_clip import CLIPModel
from src.models.clip.tokenization_clip_fast import CLIPTokenizerFast


def normalize(features):
    return features / features.norm(p=2, dim=-1, keepdim=True)


# CLIP
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel(CLIPConfig())

text_input = tokenizer(["a photo of a cat", "a image of a dog"], padding=True, return_tensors="pt")
text_feature = model.get_text_features(**text_input)
text_feature = normalize(text_feature)

image_input = dict(pixel_values=torch.rand(2, 3, 224, 224))
image_feature = model.get_image_features(**image_input)
image_feature = normalize(image_feature)

logits_per_text = torch.matmul(text_feature, image_feature.t())
logits_per_image = logits_per_text.t()
print(logits_per_image)
