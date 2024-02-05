import os

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from . import VLMClassificationDataset, UCF101_CLASS_NAME, UCF101_PROMPT


class UCF101(VLMClassificationDataset, Dataset):
    dataset_path = 'UCF-101-midframes'
    n_class = 101

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = ImageFolder(os.path.join(root, self.dataset_path))
        class_name_list = UCF101_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return UCF101_PROMPT
