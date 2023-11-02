from .create_loader2 import create_loader_v2
from .dataset import *

DATASET_DICT = {
    'imagenetra': ImageNetRandaugPrompt,
    'imagenet': ImageNet,
    'imagenet_a': ImageNetA,
    'imagenet_r': ImageNetR,
    'imagenet_v2': ImageNetV2,
    'imagenet_sketch': ImageNetSketch,
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
}


def create_dataset(ds_cfg, **kwargs):
    ds_kwargs = dict(
        transform=kwargs.get('transform', None),
        root=kwargs.get('root', ds_cfg.root),
        target_transform=kwargs.get('target_transform', None),
        n_shot=kwargs.get('n_shot', 0),
    )
    if kwargs.get('split', None):
        ds_kwargs['split'] = kwargs['split']
    return DATASET_DICT[ds_cfg.name](**ds_kwargs)


def create_dataloader(cfg, dataset, is_train):
    aug = cfg.dataset.augmentation

    loader = create_loader_v2(
        dataset,
        input_size=tuple(cfg.dataset.train_size) if is_train else tuple(cfg.dataset.eval_size),
        batch_size=cfg.train.batch_size,
        is_training=is_train,
        use_prefetcher=aug.prefetcher,
        no_aug=aug.no_aug,
        re_prob=aug.re_prob,
        re_mode=aug.re_mode,
        re_count=aug.re_count,
        re_split=aug.re_split,
        scale=aug.scale,
        ratio=aug.ratio,
        hflip=aug.hflip,
        vflip=aug.vflip,
        color_jitter=aug.color_jitter,
        auto_augment=aug.auto_aug,
        num_aug_repeats=aug.aug_repeats,
        num_aug_splits=aug.aug_splits,
        interpolation=aug.train_interpolation if is_train else aug.test_interpolation,
        mean=tuple(aug.mean),
        std=tuple(aug.std),
        num_workers=cfg.train.num_workers,
        distributed=cfg.distributed,
        pin_memory=aug.pin_mem,
        use_multi_epochs_loader=aug.use_multi_epochs_loader,
        worker_seeding=aug.worker_seeding,
    )

    return loader


if __name__ == '__main__':
    for k, v in DATASET_DICT.items():
        ds = v('/data/vlm')
        if len(ds.class_name) != ds.n_class:
            print(k, len(ds.class_name), ds.n_class)
