import os

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from . import VLMClassificationDataset, CALTECH101_CLASS_NAME, CALTECH101_PROMPT


class Caltech101(VLMClassificationDataset, Dataset):
    dataset_path = 'caltech101'
    n_class = 101

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        dataset = ImageFolder(os.path.join(root, self.dataset_path, split))
        super().__init__(root, *self._imgs_targets(dataset), CALTECH101_CLASS_NAME, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return CALTECH101_PROMPT
