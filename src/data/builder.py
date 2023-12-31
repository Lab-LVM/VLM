from functools import partial

from timm.data import create_transform as timm_create_transform
from torch.utils.data import DataLoader

from src.data.dataset import *

DATASET_DICT = {
    'imagenetra': ImageNetRandaugPromptFeatures,  # ImageNetRandaugPrompt
    'imagenetraV2': ImageNetRandaugPromptFeaturesV2,
    'imagenetram9': partial(ImageNetRandaugPromptFeatures, dataset_path='m9_imageNet_train'),
    'imagenetraText': ImageNetRandaugPrompt,
    'imagenetra2': ImageNetRandaugPromptV2,
    'imagenetsim': ImageNetSimplePrompt,
    'imagenet': partial(ImageNetEvalFeatures, dataset_name='imagenet'),
    'imagenet_a': partial(ImageNetEvalFeatures, dataset_name='imagenet_a'),
    'imagenet_r': partial(ImageNetEvalFeatures, dataset_name='imagenet_r'),
    'imagenet_v2': partial(ImageNetEvalFeatures, dataset_name='imagenet_v2'),
    'imagenet_sketch': partial(ImageNetEvalFeatures, dataset_name='imagenet_sketch'),
    'objectnet': partial(ImageNetEvalFeatures, dataset_name='objectnet'),
    # 'imagenet': ImageNet,
    # 'imagenet_a': ImageNetA,
    # 'imagenet_r': ImageNetR,
    # 'imagenet_v2': ImageNetV2,
    # 'imagenet_sketch': ImageNetSketch,
    # 'objectnet': ObjectNet,
    'caltech101': Caltech101,
    'eurosat': EuroSAT,
    'fgvc': FGVCAircraft,
    'flowers102': Flowers102,
    'food101': Food101,
    'oxfordiiitpet': OxfordIIITPet,
    'stanfordcars': StanfordCars,
    'sun397': SUN397,
    'dtd': DescribableTextures,
    'ucf101': UCF101,
    'cifar100': CIFAR100,
    'pcam': PCam,
    'country211': Country211,
}


def create_dataset(ds_cfg, is_train, **kwargs):
    ds_kwargs = dict(
        transform=kwargs.get('transform', create_transform(ds_cfg, is_train)),
        root=kwargs.get('root', ds_cfg.root),
        target_transform=kwargs.get('target_transform', None),
        n_shot=kwargs.get('n_shot', 0),
    )
    if kwargs.get('split', None):
        ds_kwargs['split'] = kwargs['split']
    if kwargs.get('backbone', None):
        ds_kwargs['backbone'] = kwargs['backbone']

    return DATASET_DICT[ds_cfg.name](**ds_kwargs)


def create_dataloader(cfg, dataset, is_train):
    loader = DataLoader(dataset, cfg.train.batch_size, shuffle=is_train, num_workers=cfg.train.num_workers,
                        drop_last=is_train, pin_memory=True)
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
    import omegaconf

    for k, v in DATASET_DICT.items():
        cfg = omegaconf.OmegaConf.load(f'/home/seungmin/dmount/VLM/configs/dataset/{k}.yaml')
        cfg.root = '/data'
        if cfg.train is not None:
            ds = v(cfg.root, cfg.train)
            ds_val = v(cfg.root, cfg.valid)

            print(f'{ds.__class__.__name__}: {len(ds.imgs)} / {len(ds_val.imgs)}')
        else:
            ds = v(cfg.root)
            print(f'NOSPLIT {ds.__class__.__name__}: {len(ds.imgs)}')
