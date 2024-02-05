from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import StanfordCars as TorchStanfordCars

from . import VLMClassificationDataset, STANFORDCARS_CLASS_NAME, STANFORDCARS_PROMPT


class StanfordCars(VLMClassificationDataset, Dataset):
    dataset_path = 'stanford_cars'
    n_class = 196

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchStanfordCars(root, split)
        super().__init__(root, *self._imgs_targets(dataset), STANFORDCARS_CLASS_NAME, transform, target_transform,
                         n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs, targets = list(), list()
        for i in range(len(dataset)):
            _data = dataset._samples[i]
            imgs.append(_data[0])
            targets.append(_data[1])
        return imgs, targets

    @property
    def prompt(self):
        return STANFORDCARS_PROMPT

    def _data_dict(self):
        train_dataset = TorchStanfordCars(self.root, 'train')
        train_data_dict = defaultdict(list)
        _imgs, _targets = self._imgs_targets(train_dataset)

        for i in range(len(_imgs)):
            train_data_dict[_targets[i]].append(str(_imgs[i]))
        return train_data_dict
