import os
from collections import OrderedDict
from pathlib import Path
from typing import List

import hydra.utils
from hydra import compose
from omegaconf import ListConfig, OmegaConf
from termcolor import colored

from ..data import DATASET_DICT

VLZB = ['caltech101', 'eurosat', 'fgvc', 'flowers102', 'food101', 'oxfordiiitpet', 'stanfordcars', 'sun397', 'dtd',
        'ucf101', 'pcam', 'country211', 'imagenet', 'cifar100']

IMAGENET_DS = ['imagenet', 'imagenet_r', 'imagenet_a', 'imagenet_v2', 'imagenet_sketch', 'objectnet']


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


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

    for k in dataset_list:
        ds_dict[k] = compose(os.path.join('dataset', k.replace('_full', ''))).dataset
        ds_dict[k]['train_size'] = cfg.train_size
        ds_dict[k]['eval_size'] = cfg.eval_size
    return ds_dict


def to_list(item):
    if isinstance(item, (List, ListConfig)):
        return item
    return [item]


def filter_grad(model, adapter_lr=None):
    backbone = []
    adapter = []
    adapter_name = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'adapter' in name:
            adapter_name.append(name)
            adapter.append(param)
        else:
            backbone.append(param)
    params = [{'params': adapter}, {'params': backbone}]
    if adapter_lr is not None:
        params[0]['lr'] = adapter_lr
    return params

    # return filter(lambda p: p.requires_grad, model.parameters())


def import_config(cfg):
    assert cfg.checkpoint and cfg.import_cfg, f'Checkpoint and import_cfg isn\'t True. Now, {cfg.checkpoint}, {cfg.import_cfg}'

    path = Path(hydra.utils.get_original_cwd()) / cfg.checkpoint

    weight_file = path / 'best.ckpt'
    cfg_file = path / '.hydra/config.yaml'
    override_file = Path(os.getcwd()) / '.hydra/overrides.yaml'

    import_cfg = OmegaConf.load(cfg_file)
    override_cfg = list(OmegaConf.load(override_file))

    for value in override_cfg:
        key, value = value.split('=')
        OmegaConf.update(import_cfg, key, OmegaConf.select(cfg, key))

    import_cfg.checkpoint = weight_file
    import_cfg.import_cfg = True
    import_cfg.full_dataset_name = import_cfg.dataset.name
    OmegaConf.set_struct(import_cfg, True)

    print(f'{colored("[Notice] Config is changed by", "green")} {cfg.checkpoint}.')
    return import_cfg


def move_dir(cfg):
    working_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    working_time = working_dir.name.rsplit('_', 1)[-1]

    root = working_dir.parent

    dataset_name = cfg.get('full_dataset_name', "NONE")
    moving_dir = root / f'{dataset_name}_{cfg.name}_{working_time}'
    os.rename(working_dir, moving_dir)


class EmptyScheduler:
    def step(self, *args, **kwargs):
        pass

    def step_update(self, *args, **kwargs):
        pass

    def state_dict(self, *args, **kwargs):
        return None
