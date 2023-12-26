from typing import MutableMapping, List

from hydra import initialize, compose
from omegaconf import ListConfig
from timm import create_model

WIDTH = 105


def format_bytes(size):
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power:
        size /= power
        n += 1
    return f'{size:.3f} {power_labels[n]}'


def count_parameters(model):
    return format_bytes(sum(p.numel() for p in model.parameters() if p.requires_grad))


def flatten_dict(d: MutableMapping) -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v).items())
        else:
            items.append((new_key, v))
    return dict(items)


def print_tabular(title, table):
    column = 2

    title_space = int((WIDTH - len(title)) / 2)
    print('-' * WIDTH)
    print(' ' * title_space + title)
    print('-' * WIDTH)

    for i, (key, value) in enumerate(table.items()):
        if key == 'eval_metrics':
            value = [s[0] for s in value]
        if isinstance(value, (List, ListConfig)):
            value = ','.join(str(v) for v in value)
        if value is None:
            value = 0.0

        is_final = (i + 1) % column == 0
        string = f'|{key:^24} : {value:^24}'
        print(string, end='|\n' if is_final else '')


def print_meta_data(cfg, model=None, train_loader=None, val_loader=None):
    title = cfg.info.project.replace('-', ' ')
    contents = {
        'ID': cfg.info.id,
        'Entity': cfg.info.entity,
        'Experiment': cfg.name,
        'Model': cfg.model.model_name,
        'Dataset': cfg.dataset.name,
        'Task': cfg.dataset.task,
        '# of Parameter': count_parameters(model) if model is not None else None,
        '# of Class': cfg.dataset.num_classes,
        '# of Train': len(train_loader.dataset) if train_loader is not None else None,
        '# of Eval': len(val_loader.dataset) if val_loader is not None else None,
    }
    print_tabular(title, contents)

    title = 'Experiment Setting'
    contents = {
        'Epochs': cfg.train.epochs,
        'Batch Size': f'{cfg.train.batch_size}',# / {cfg.train.total_batch}',
        'World Size': cfg.world_size,
        # 'Grad Accum': cfg.train.optimizer.get('grad_accumulation', 1),
        'Optimizer': cfg.train.optimizer.get('opt', 'LBFGS'),
        'LR': cfg.train.optimizer.lr,
        'Weight Decay': cfg.train.optimizer.get('weight_decay', 0),
        'Scheduler': cfg.train.scheduler.sched,
        'CoolDown Epochs': cfg.train.scheduler.cooldown_epochs,
        'Warmup LR': cfg.train.scheduler.warmup_lr,
        'Warmup Epochs': cfg.train.scheduler.warmup_epochs,
        'Min LR': cfg.train.scheduler.min_lr,
        'Criterion': cfg.train.criterion,
    }
    print_tabular(title, contents)

    title = 'Augmentations'
    contents = {
        'Train Size': cfg.dataset.train_size,
        'Eval Size': cfg.dataset.eval_size,
        'Auto Aug': cfg.dataset.augmentation.auto_aug,
        'Interpolation': f'{cfg.dataset.augmentation.train_interpolation} / {cfg.dataset.augmentation.test_interpolation}',
        'Smoothing': cfg.dataset.augmentation.smoothing,
        'Cutmix': cfg.dataset.augmentation.cutmix,
        'Mixup': cfg.dataset.augmentation.mixup,
        'Crop pct': cfg.dataset.augmentation.crop_pct,
        'Aug Repeats': cfg.dataset.augmentation.aug_repeats,
        'Re Prob': cfg.dataset.augmentation.re_prob,
    }
    print_tabular(title, contents)
    print('-' * WIDTH)


if __name__ == '__main__':
    with initialize(config_path='../../configs', version_base='1.3'):
        cfg = compose('config.yaml', overrides=['model.model_name=resnet51'])
        cfg.name = cfg.model.model_name

        print_meta_data(cfg, create_model('resnet18'))
