from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import SUN397 as TorchSUN397

from . import VLMClassificationDataset, SUN397_CLASS_NAME, SUN397_PROMPT


class SUN397(VLMClassificationDataset, Dataset):
    dataset_path = 'SUN397'
    n_class = 397

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = TorchSUN397(root)
        self.dataset = dataset
        super().__init__(root, dataset._image_files, dataset._labels, SUN397_CLASS_NAME, transform, target_transform,
                         n_shot)

    @property
    def prompt(self):
        return SUN397_PROMPT

    def _data_dict(self):
        data_dict = defaultdict(list)
        for i in range(len(self.dataset._image_files)):
            data_dict[self.dataset._labels[i]].append(str(self.dataset._image_files[i]))
        return data_dict
