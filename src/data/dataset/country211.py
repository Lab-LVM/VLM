import os
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import Country211 as TorchCountry211

from . import VLMClassificationDataset, COUNTRY211_CLASS_NAME, COUNTRY211_PROMPT


class Country211(VLMClassificationDataset, Dataset):
    dataset_path = 'country211'
    n_class = 211

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchCountry211(root, split)
        super().__init__(root, *self._imgs_targets(dataset), COUNTRY211_CLASS_NAME, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return COUNTRY211_PROMPT
