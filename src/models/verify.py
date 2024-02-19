import datasets
import torch
from transformers import CLIPTokenizerFast, CLIPModel

datasets.disable_progress_bar()


def normalize(features):
    return features / features.norm(p=2, dim=-1, keepdim=True)


model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32', num_proc=5)

# dataset = datasets.Dataset.from_dict({'text': ["a photo of a cat", "a image of a dog"]})
# text_input = dataset.map(lambda item: tokenizer(item['text'], padding='max_length', return_attention_mask=False),
#                          remove_columns=['text'], batched=True).with_format('pt')['input_ids']

text = ["a photo of a cat", "a image of a dog"]
text_input = tokenizer(text, padding='max_length', return_attention_mask=False, return_tensors='pt')['input_ids']
print(text_input)

text_feature = model.text_projection(model.text_model(text_input)[1])
text_feature = normalize(text_feature)

image_input = torch.rand(2, 3, 224, 224)
image_feature = model.visual_projection(model.vision_model(image_input)[1])
image_feature = normalize(image_feature)

logits_per_text = torch.matmul(text_feature, image_feature.t()) * model.logit_scale.exp()
logits_per_image = logits_per_text.t()
print(logits_per_image)

print(text_feature.shape, image_feature.shape)

# All in one
out = model(pixel_values=image_input, input_ids=text_input)
print(out.keys())
