from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import Food101 as TorchFood101
from torchvision.transforms import transforms

from . import VLMClassificationDataset, FOOD101_CLASS_NAME, FOOD101_PROMPT


class Food101(VLMClassificationDataset, Dataset):
    dataset_path = 'food-101'
    n_class = 101

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchFood101(root, split)
        class_name_list = FOOD101_CLASS_NAME
        super().__init__(root, dataset._image_files, dataset._labels, class_name_list, transform, target_transform,
                         n_shot)

    @property
    def prompt(self):
        return FOOD101_PROMPT

    def _data_dict(self):
        train_dataset = TorchFood101(self.root, 'train')
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset._image_files)):
            train_data_dict[train_dataset._labels[i]].append(str(train_dataset._image_files[i]))
        return train_data_dict
