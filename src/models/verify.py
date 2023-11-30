import datasets
import torch
from transformers import CLIPTokenizerFast
from src.models import *
from src.utils.registry import create_model

datasets.disable_progress_bar()


def normalize(features):
    return features / features.norm(dim=-1, keepdim=True)


model = create_model('CLIP')
# tokenizer = create_model('CLIP_tokenizer')
tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32', num_proc=5)

# dataset = datasets.Dataset.from_dict({'text': ["a photo of a cat", "a image of a dog"]})
# text_input = dataset.map(lambda item: tokenizer(item['text'], padding='max_length', return_attention_mask=False),
#                          remove_columns=['text'], batched=True).with_format('pt')['input_ids']

text = ["a photo of a cat", "a image of a dog"]
text_input = tokenizer(text, padding='max_length', return_attention_mask=False, return_tensors='pt')['input_ids']
print(text_input)

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
