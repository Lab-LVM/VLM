import torch
import src.models
from src.utils.registry import create_model

def normalize(features):
    return features / features.norm(dim=-1, keepdim=True)


model = create_model('CLIP')
tokenizer = create_model('CLIP_tokenizer')

text_input = tokenizer(["a photo of a cat", "a image of a dog"])
print(text_input.shape)
text_feature = model.encode_text(text_input)
text_feature = normalize(text_feature)

image_input = torch.rand(2, 3, 224, 224)
image_feature = model.encode_image(image_input)
image_feature = normalize(image_feature)

logits_per_text = torch.matmul(text_feature, image_feature.t())
logits_per_image = logits_per_text.t()
print(logits_per_image)

print(text_feature.shape, image_feature.shape)

out = model(image_input, text_input)
