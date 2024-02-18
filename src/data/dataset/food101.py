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
