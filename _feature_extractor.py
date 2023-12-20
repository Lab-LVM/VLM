import os
import pickle
from pathlib import Path

import torch
from hydra import initialize, compose
from tqdm import tqdm

from src.data import create_dataloader
from src.data.dataset import ImageNetRandaugPromptText
from src.models import CLIPTMP, CLIP_tokenizer

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '8'


def create_dataset(ds_cfg, **kwargs):
    ds_kwargs = dict(
        transform=kwargs.get('transform', None),
        root=kwargs.get('root', ds_cfg.root),
        target_transform=kwargs.get('target_transform', None),
        n_shot=kwargs.get('n_shot', 0),
    )
    if kwargs.get('split', None):
        ds_kwargs['split'] = kwargs['split']

    return ImageNetRandaugPromptText(**ds_kwargs)


if __name__ == '__main__':
    with initialize('configs', version_base='1.3'):
        cfg = compose('train_config', overrides=['model.backbone=ViT-B16', '+setup=clip_simple_adapter',
                                                 'dataset.augmentation.prefetcher=False'])
    cfg.train.batch_size = 1024

    device = torch.device('cuda')

    clip = CLIPTMP(**cfg.model)
    clip.eval()
    clip.to(device)

    tokenizer = CLIP_tokenizer()

    ds = create_dataset(cfg.dataset, split=cfg.dataset.train, n_shot=0)
    dl = create_dataloader(cfg, ds, is_train=True)
    dl.dataset.setup_prompt_transform()
    root = Path(f'/home/seungmin/shared/hdd_ext/hdd4000/seungmin/imageNet_train_features_B16')
    root.mkdir(exist_ok=True)

    keys = ('vision_features', 'language_features', 'targets')

    for i in range(39, 55):
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
