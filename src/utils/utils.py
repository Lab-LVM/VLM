import os
from collections import OrderedDict
from typing import List

from hydra import compose
from omegaconf import ListConfig

from ..data import DATASET_DICT

VLZB = ['caltech101', 'eurosat', 'fgvc', 'flowers102', 'food101', 'oxfordiiitpet', 'stanfordcars', 'sun397', 'dtd',
        'ucf101', 'pcam', 'imagenet', 'cifar100']

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
        ds_dict[k] = compose(os.path.join('dataset', k)).dataset
        ds_dict[k]['train_size'] = cfg.train_size
        ds_dict[k]['eval_size'] = cfg.eval_size
    return ds_dict


def to_list(item):
    if isinstance(item, (List, ListConfig)):
        return item
    return [item]


def filter_grad(model):
    return filter(lambda p: p.requires_grad, model.parameters())


class EmptyScheduler:
    def step(self, *args, **kwargs):
        pass

    def step_update(self, *args, **kwargs):
        pass

    def state_dict(self, *args, **kwargs):
        return None
