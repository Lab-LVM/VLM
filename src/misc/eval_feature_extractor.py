import os
import pickle
from pathlib import Path

import torch
from hydra import initialize, compose
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.models.clip as clip
from src.data import DATASET_DICT, create_dataset
from src.models import CLIP_tokenizer
from src.utils import VLZB, IMAGENET_DS

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


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


def _save_file(obj, file_name):
    torch.save(obj, file_name)


@torch.no_grad()
def build_text_classifier(model, train_dataset, tokenizer, device):
    text_classifier = list()
    model.eval()
    for class_name in tqdm(train_dataset.class_name, desc='Build Text Classifier'):
        text = [p(class_name) for p in train_dataset.prompt]
        text_input = tokenizer(text, padding='max_length', return_attention_mask=False, return_tensors='pt')[
            'input_ids'].to(device)

        with torch.cuda.amp.autocast():
            text_feature = model.encode_text(text_input)
        text_classifier.append(text_feature)

    text_classifier = torch.stack(text_classifier, dim=0).to(device)
    return text_classifier


@torch.no_grad()
def build_query_set(cfg, model, dataset, device):
    features, labels = list(), list()
    is_train = False
    loader = DataLoader(dataset, cfg.train.batch_size, shuffle=is_train, num_workers=cfg.train.num_workers,
                        drop_last=is_train, pin_memory=True)

    model.eval()
    for data in tqdm(loader, total=len(loader), desc=f'Build Query Set'):
        x, y = map(lambda x: x.to(device), data)
        x = x.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast():
            image_features = model.encode_image(x)

        features.append(image_features.detach().cpu())
        labels.append(y.detach().cpu())

    qry_features = torch.cat(features, dim=0)
    qry_labels = torch.cat(labels)

    return qry_features, qry_labels


def dataset2dict(cfg):
    ds_dict = dict()

    if cfg.name == 'all':
        dataset_list = list(DATASET_DICT.keys())
    elif cfg.name == 'vlzb':
        dataset_list = VLZB
    elif cfg.name == 'imagenet_ds':
        dataset_list = IMAGENET_DS
    else:
        return {cfg.name: cfg}

    with initialize('../../configs', version_base='1.3'):
        for k in dataset_list:
            ds_dict[k] = compose(os.path.join('dataset', k)).dataset
            ds_dict[k]['train_size'] = cfg.train_size
            ds_dict[k]['eval_size'] = cfg.eval_size
    return ds_dict


if __name__ == '__main__':
    with initialize('../../configs', version_base='1.3'):
        cfg = compose('train_config',
                      overrides=['model.backbone=ViT-L14@336px', '+setup=our', 'dataset.augmentation.prefetcher=False',
                                 'dataset=imagenet_ds'])
    cfg.train.batch_size = 512
    cfg.dataset.name = 'imagenet_ds'

    device = torch.device('cuda')

    clip = CLIPTMP(cfg.model.backbone)
    clip.eval()
    clip.to(device)

    tokenizer = CLIP_tokenizer()
    root = Path(f'/home/seungmin/dmount/feature_data/L14@336px_imagenet_ds_eval')
    root.mkdir(exist_ok=True, parents=True)
    backbone = cfg.model.backbone.split('-')[-1]
    for k, v in dataset2dict(cfg.dataset).items():
        print(k)
        cfg.dataset = v
        cfg.dataset.train_size = [3, 336, 336]
        cfg.dataset.eval_size = [3, 336, 336]
        train_dataset = create_dataset(cfg.dataset, is_train=True, split=cfg.dataset.train)
        test_dataset = create_dataset(cfg.dataset, is_train=False, split=cfg.dataset.test)

        text_feature = build_text_classifier(clip, train_dataset, tokenizer, device)
        image_feature, labels = build_query_set(cfg, clip, test_dataset, device)

        obj = dict()
        obj['vision_features'] = image_feature.detach().cpu()
        obj['language_features'] = text_feature.detach().cpu()
        obj['targets'] = labels.detach().cpu()

        with open(root / f'{k}.pkl', 'wb') as f:
            pickle.dump(obj, f)
