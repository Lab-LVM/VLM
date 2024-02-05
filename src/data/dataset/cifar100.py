from collections import defaultdict

import numpy as np
from PIL.Image import fromarray
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100 as TorchCIFAR100

from . import VLMClassificationDataset, CIFAR100_CLASS_NAME, CIFAR_PROMPT


class CIFAR100(VLMClassificationDataset, Dataset):
    dataset_path = 'cifar-100-python'
    n_class = 100

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchCIFAR100(root, train=split == 'train')
        super().__init__(root, dataset.data, dataset.targets, CIFAR100_CLASS_NAME, transform, target_transform, n_shot)
        self.targets = np.array(self.targets)

    @property
    def prompt(self):
        return CIFAR_PROMPT

    def _data_dict(self):
        train_dataset = TorchCIFAR100(self.root, train=True)
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset.data)):
            train_data_dict[train_dataset.targets[i]].append(str(train_dataset.data[i]))
        return train_data_dict

    @staticmethod
    def loader(data):
        return fromarray(data)
