from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import Flowers102 as TorchFlowers102

from . import VLMClassificationDataset, FLOWERS102_CLASS_NAME, FLOWERS102_PROMPT


class Flowers102(VLMClassificationDataset, Dataset):
    dataset_path = 'flowers-102'
    n_class = 102

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        if split == 'trainval':
            train_dataset = TorchFlowers102(root, 'train')
            val_dataset = TorchFlowers102(root, 'val')
            train_dataset._image_files.extend(val_dataset._image_files)
            train_dataset._labels.extend(val_dataset._labels)
            dataset = train_dataset
        else:
            dataset = TorchFlowers102(root, split)
        super().__init__(root, dataset._image_files, dataset._labels, FLOWERS102_CLASS_NAME, transform,
                         target_transform, n_shot)

    @property
    def prompt(self):
        return FLOWERS102_PROMPT

    def _data_dict(self):
        train_dataset = TorchFlowers102(self.root, 'train')
        val_dataset = TorchFlowers102(self.root, 'val')
        train_dataset._image_files.extend(val_dataset._image_files)
        train_dataset._labels.extend(val_dataset._labels)

        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset)):
            train_data_dict[train_dataset._labels[i]].append(str(train_dataset._image_files[i]))
        return train_data_dict
