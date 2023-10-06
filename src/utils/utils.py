import os
from collections import OrderedDict
from typing import List

from hydra import compose
from omegaconf import ListConfig

from ..data import DATASET_DICT


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def dataset2dict(cfg):
    if cfg.name == 'all':
        ds_dict = dict()
        for k, v in DATASET_DICT.items():
            ds_dict[k] = compose(os.path.join('dataset', k)).dataset
        return ds_dict
    return {cfg.name: cfg}


def to_list(item):
    if isinstance(item, (List, ListConfig)):
        return item
    return [item]


def filter_grad(model):
    return filter(lambda p: p.requires_grad, model.parameters())
