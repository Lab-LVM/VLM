from functools import partial

from timm.data import create_transform as timm_create_transform
from torch.utils.data import DataLoader

from src.data.dataset import *
from src.data.mixup import FastCollateMixup

DATASET_DICT = {
    'sa3': partial(ImageNetRandaugPromptFeaturesV2, dataset_path='imageNet_train_with_scaleAug3'),
    'sa5': partial(ImageNetRandaugPromptFeaturesV2, dataset_path='imageNet_train_with_scaleAug5'),
    'sa7': partial(ImageNetRandaugPromptFeaturesV2, dataset_path='imageNet_train_with_scaleAug7'),
    'sa9': partial(ImageNetRandaugPromptFeaturesV2, dataset_path='imageNet_train_with_scaleAug9'),
    'sa13': partial(ImageNetRandaugPromptFeaturesV2, dataset_path='imageNet_train_with_scaleAug13'),

    'imagenetra': ImageNetRandaugPromptFeatures,  # ImageNetRandaugPrompt
    'imagenetraV2': ImageNetRandaugPromptFeaturesV2,
    'imagenetram9': partial(ImageNetRandaugPromptFeatures, dataset_path='m9_imageNet_train'),
    'imagenetraText': ImageNetRandaugPrompt,
    'imagenetraTextV2': ImageNetRandaugPromptV2,
    'imagenetsim': ImageNetSimplePrompt,
    'imagenet': partial(ImageNetEvalFeatures, dataset_name='imagenet'),
    'imagenet_a': partial(ImageNetEvalFeatures, dataset_name='imagenet_a'),
    'imagenet_r': partial(ImageNetEvalFeatures, dataset_name='imagenet_r'),
    'imagenet_v2': partial(ImageNetEvalFeatures, dataset_name='imagenet_v2'),
    'imagenet_sketch': partial(ImageNetEvalFeatures, dataset_name='imagenet_sketch'),
    'objectnet': partial(ImageNetEvalFeatures, dataset_name='objectnet'),
    'imagenet_full': ImageNet,
    'imagenet_a_full': ImageNetA,
    'imagenet_r_full': ImageNetR,
    'imagenet_v2_full': ImageNetV2,
    'imagenet_sketch_full': ImageNetSketch,
    'objectnet_full': ObjectNet,
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
    aug = cfg.dataset.augmentation

    collate_fn = None
    mixup_active = aug.mixup > 0 or aug.cutmix > 0. or aug.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=aug.mixup, cutmix_alpha=aug.cutmix, cutmix_minmax=aug.cutmix_minmax,
            prob=aug.mixup_prob, switch_prob=aug.mixup_switch_prob, mode=aug.mixup_mode,
            label_smoothing=aug.smoothing, num_classes=dataset.num_classes)
        collate_fn = FastCollateMixup(**mixup_args)

    loader = DataLoader(dataset, cfg.train.batch_size, shuffle=is_train, num_workers=cfg.train.num_workers,
                        collate_fn=collate_fn, drop_last=is_train, pin_memory=True)
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
