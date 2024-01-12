import random
from functools import partial

import numpy as np
import torch
from timm.data import create_transform as timm_create_transform
from torch.utils.data import DataLoader

from src.data.dataset import *
from src.data.mixup import FastCollateMixup

DATASET_DICT = {
    'sa9': partial(ImageNetRandaugPromptFeatures, dataset_path='imageNet_train_with_scaleAug9'),
    'sa9o': partial(ImageNetRandaugPromptFeatures, dataset_path='imageNet_train_with_scaleAug9_withOrigin'),

    'imagenetra': ImageNetRandaugPromptFeatures,
    'imagenetraText': ImageNetRandaugPromptText,
    'imagenetraTextOri': ImageNetRandaugPromptOriginalText,
    'imagenetsimText': ImageNetSimplePromptText,

    'f_imagenet': partial(ImageNetEvalFeatures, dataset_name='imagenet'),
    'f_imagenet_a': partial(ImageNetEvalFeatures, dataset_name='imagenet_a'),
    'f_imagenet_r': partial(ImageNetEvalFeatures, dataset_name='imagenet_r'),
    'f_imagenet_v2': partial(ImageNetEvalFeatures, dataset_name='imagenet_v2'),
    'f_imagenet_sketch': partial(ImageNetEvalFeatures, dataset_name='imagenet_sketch'),
    'f_objectnet': partial(ImageNetEvalFeatures, dataset_name='objectnet'),

    'imagenet': ImageNet,
    'imagenet_a': ImageNetA,
    'imagenet_r': ImageNetR,
    'imagenet_v2': ImageNetV2,
    'imagenet_sketch': ImageNetSketch,
    'objectnet': ObjectNet,

    'caltech101': Caltech101,
    'eurosat': EuroSAT,
    'fgvc': FGVCAircraft,
    'flowers102': Flowers102Text,
    'food101': Food101,
    'oxfordiiitpet': OxfordIIITPet,
    'stanfordcars': StanfordCars,
    'sun397': SUN397,
    'dtd': DescribableTextures,
    'ucf101': UCF101,
    'cifar100': CIFAR100Text,
    'pcam': PCamText,
    'country211': Country211,
}


def create_dataset(ds_cfg, is_train, **kwargs):
    ds_kwargs = dict(
        transform=kwargs.get('transform', create_transform(ds_cfg, is_train)),
        root=kwargs.get('root', ds_cfg.root),
        target_transform=kwargs.get('target_transform', None),
        n_shot=kwargs.get('n_shot', 0),
        is_train=is_train,
    )
    if kwargs.get('split', None):
        ds_kwargs['split'] = kwargs['split']
    if kwargs.get('backbone', None):
        ds_kwargs['backbone'] = kwargs['backbone']

    return DATASET_DICT[ds_cfg.name](**ds_kwargs)


def to_ndarray(dataset):
    if isinstance(dataset.imgs, list):
        dataset.imgs = np.array(dataset.imgs)
    if isinstance(dataset.targets, list):
        dataset.targets = np.array(dataset.targets)
    return dataset


def fill_drop_last(dataset, batch_size, world_size):
    total_batch_size = batch_size * world_size
    fill_size = total_batch_size - (len(dataset) % total_batch_size)

    dataset = to_ndarray(dataset)

    if isinstance(dataset.imgs, torch.Tensor):
        rand_idx = torch.randperm(len(dataset))[:fill_size]
        fill_img, fill_target = dataset.imgs[rand_idx], dataset.targets[rand_idx]
        dataset.imgs = torch.cat([dataset.imgs, fill_img])
        dataset.targets = torch.cat([dataset.targets, fill_target])
    elif isinstance(dataset.imgs, np.ndarray):
        rand_idx = np.random.choice(len(dataset), fill_size, replace=False)
        fill_img, fill_target = dataset.imgs[rand_idx], dataset.targets[rand_idx]
        dataset.imgs = np.concatenate([dataset.imgs, fill_img])
        dataset.targets = np.concatenate([dataset.targets, fill_target])
    else:
        fill_img, fill_target = zip(*random.sample(list(zip(dataset.imgs, dataset.targets)), fill_size))
        dataset.imgs.extend(fill_img)
        dataset.targets.extend(fill_target)

    return dataset


def create_dataloader(cfg, dataset, is_train, fill_last=True):
    aug = cfg.dataset.augmentation

    if is_train and fill_last:
        dataset = fill_drop_last(dataset, cfg.train.batch_size, cfg.world_size)

    collate_fn = None
    mixup_active = aug.mixup > 0 or aug.cutmix > 0. or aug.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=aug.mixup, cutmix_alpha=aug.cutmix, cutmix_minmax=aug.cutmix_minmax,
            prob=aug.mixup_prob, switch_prob=aug.mixup_switch_prob, mode=aug.mixup_mode,
            label_smoothing=aug.smoothing, num_classes=dataset.num_classes)
        collate_fn = FastCollateMixup(**mixup_args)

    loader = DataLoader(dataset, cfg.train.batch_size, shuffle=is_train, num_workers=cfg.train.num_workers,
                        collate_fn=collate_fn, drop_last=False, pin_memory=True)
    return loader


def create_transform(ds_cfg, is_train):
    aug = ds_cfg.augmentation
    return timm_create_transform(
        tuple(ds_cfg.train_size) if is_train else tuple(ds_cfg.eval_size),
        is_training=is_train,
        no_aug=aug.no_aug,
        scale=aug.scale,
        ratio=aug.ratio,
        hflip=aug.hflip,
        vflip=aug.vflip,
        color_jitter=aug.color_jitter,
        auto_augment=aug.auto_aug,
        interpolation=aug.train_interpolation if is_train else aug.test_interpolation,
        mean=tuple(aug.mean),
        std=tuple(aug.std),
        crop_pct=aug.crop_pct,
        re_prob=aug.re_prob,
        re_mode=aug.re_mode,
        re_count=aug.re_count,
    )


if __name__ == '__main__':
    from hydra import initialize, compose

    for k, v in DATASET_DICT.items():
        if k != 'imagenetraText':
            continue
        k = 'imagenet'
        with initialize('../../configs', version_base='1.3'):
            cfg = compose('train_config', overrides=['dataset=imagenet'])
        ds_cfg = cfg.dataset
        ds_cfg.root = '/data'
        ds_cfg.augmentation.cutmix = 1.0
        # ds_cfg.augmentation.mixup = 1.0
        # if ds_cfg.train is not None:
        #     ds = v(ds_cfg.root, ds_cfg.train)
        #     ds_val = v(ds_cfg.root, ds_cfg.valid)
        #
        #     print(f'{ds.__class__.__name__}: {len(ds.imgs)} / {len(ds_val.imgs)}')
        # else:
        #     ds = v(ds_cfg.root)
        #     print(f'NOSPLIT {ds.__class__.__name__}: {len(ds.imgs)}')

        cfg.dataset.name = 'imagenetraText'
        ds = create_dataset(cfg.dataset, is_train=True)
        ds.setup_prompt_transform()
        cfg.dataset = ds_cfg
        cfg.train.batch_size = 10
        dl = create_dataloader(cfg, ds, is_train=True)
