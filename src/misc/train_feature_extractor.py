import os
import pickle
from pathlib import Path

import torch
from hydra import initialize, compose
from tqdm import tqdm

from src.data import create_dataloader
from src.data.builder import create_transform
from src.data.dataset import ImageNetRandaugPromptText
from src.models import CLIP_tokenizer
import src.models.clip as clip

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def forward_for_feature_extraction(self, image, text):
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)
    return image_features, text_features


def CLIPTMP(backbone):
    model, _ = clip.load(backbone)

    for name, param in model.named_parameters():
        param.requires_grad = False

    forward_bound_method = forward_for_feature_extraction.__get__(model, model.__class__)
    setattr(model, 'forward', forward_bound_method)

    return model


def create_dataset(ds_cfg, **kwargs):
    ds_kwargs = dict(
        transform=kwargs.get('transform', create_transform(ds_cfg, True)),
        root=kwargs.get('root', ds_cfg.root),
        target_transform=kwargs.get('target_transform', None),
        n_shot=kwargs.get('n_shot', 0),
    )
    if kwargs.get('split', None):
        ds_kwargs['split'] = kwargs['split']

    return ImageNetRandaugPromptText(**ds_kwargs)


if __name__ == '__main__':
    with initialize('../../configs', version_base='1.3'):
        cfg = compose('train_config', overrides=['model.backbone=ViT-B32', '+setup=our',
                                                 'dataset.augmentation.prefetcher=False'])
    cfg.train.batch_size = 1024
    print(cfg.model.backbone)

    device = torch.device('cuda')

    clip = CLIPTMP(cfg.model.backbone)
    clip.train()
    clip.to(device)

    tokenizer = CLIP_tokenizer()

    ds = create_dataset(cfg.dataset, split=cfg.dataset.train, n_shot=0, is_train=True)
    dl = create_dataloader(cfg, ds, is_train=True)
    dl.dataset.setup_prompt_transform()
    root = Path(f'/home/seungmin/dmount/feature_data/B32_imageNet_train')
    root.mkdir(exist_ok=True)

    keys = ('vision_features', 'language_features', 'targets')

    with torch.cuda.amp.autocast():
        for i in range(0, 11):
            print(f'EPOCH: {i}')
            obj = {k: list() for k in keys}

            with torch.no_grad():
                for e, (x, y, prompt) in tqdm(enumerate(dl), total=len(dl)):
                    x = x.to(device)
                    y = y.to(device)
                    prompt = tokenizer(prompt, padding='max_length', return_attention_mask=False, return_tensors='pt')[
                        'input_ids'].to(device)

                    im, te = clip(x, prompt)
                    obj['vision_features'].append(im)
                    obj['language_features'].append(te)
                    obj['targets'].append(y)

            for k in keys:
                obj[k] = torch.concat(obj[k], dim=0).detach().cpu()

            with open(root / f'{i}.pkl', 'wb') as f:
                pickle.dump(obj, f)
