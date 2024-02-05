import os
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import ImageNet as TorchImagenet

from . import VLMClassificationDataset, IMAGENET_CLASS_NAME, IMAGENET_PROMPT


class ImageNet(VLMClassificationDataset, Dataset):
    dataset_path = 'imageNet'
    n_class = 1000

    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0, is_train=False):
        dataset = TorchImagenet(os.path.join(root, self.dataset_path), split)
        super().__init__(root, *self._imgs_targets(dataset), IMAGENET_CLASS_NAME, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return IMAGENET_PROMPT

    def _data_dict(self):
        train_dataset = TorchImagenet(os.path.join(self.root, self.dataset_path), 'train')
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset.imgs)):
            train_data_dict[train_dataset.targets[i]].append(train_dataset.imgs[i][0])

        return train_data_dict
