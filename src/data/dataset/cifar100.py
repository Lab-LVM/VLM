import random
from collections import defaultdict

import numpy as np
from PIL.Image import fromarray
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100 as TorchCIFAR100
from torchvision.transforms import transforms

from . import VLMDataset, CIFAR100_CLASS_NAME
from .ra_text import RandAugment, RAND_AUG_TRANSFORMS

ORIGINAL_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'a blurry photo of a {name}.',
    lambda name: f'a black and white photo of a {name}.',
    lambda name: f'a low contrast photo of a {name}.',
    lambda name: f'a high contrast photo of a {name}.',
    lambda name: f'a bad photo of a {name}.',
    lambda name: f'a good photo of a {name}.',
    lambda name: f'a photo of a small {name}.',
    lambda name: f'a photo of a big {name}.',
    lambda name: f'a photo of the {name}.',
    lambda name: f'a blurry photo of the {name}.',
    lambda name: f'a black and white photo of the {name}.',
    lambda name: f'a low contrast photo of the {name}.',
    lambda name: f'a high contrast photo of the {name}.',
    lambda name: f'a bad photo of the {name}.',
    lambda name: f'a good photo of the {name}.',
    lambda name: f'a photo of the small {name}.',
    lambda name: f'a photo of the big {name}.',
]
AUGMENT_PROMPT = [
    lambda augment, name: f'a {augment} photo of a {name}.',
    lambda augment, name: f'a blurry {augment} photo of a {name}.',
    lambda augment, name: f'a black and white {augment} photo of a {name}.',
    lambda augment, name: f'a low contrast {augment} photo of a {name}.',
    lambda augment, name: f'a high contrast {augment} photo of a {name}.',
    lambda augment, name: f'a bad {augment} photo of a {name}.',
    lambda augment, name: f'a good {augment} photo of a {name}.',
    lambda augment, name: f'a {augment} photo of a small {name}.',
    lambda augment, name: f'a {augment} photo of a big {name}.',
    lambda augment, name: f'a {augment} photo of the {name}.',
    lambda augment, name: f'a blurry {augment} photo of the {name}.',
    lambda augment, name: f'a black and white {augment} photo of the {name}.',
    lambda augment, name: f'a low contrast {augment} photo of the {name}.',
    lambda augment, name: f'a high contrast {augment} photo of the {name}.',
    lambda augment, name: f'a bad {augment} photo of the {name}.',
    lambda augment, name: f'a good {augment} photo of the {name}.',
    lambda augment, name: f'a {augment} photo of the small {name}.',
    lambda augment, name: f'a {augment} photo of the big {name}.',
]


class CIFAR100(VLMDataset, Dataset):
    dataset_path = 'cifar-100-python'
    n_class = 100

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchCIFAR100(root, train=split == 'train')
        class_name_list = CIFAR100_CLASS_NAME
        super().__init__(root, dataset.data, dataset.targets, class_name_list, transform, target_transform, n_shot)
        self.targets = np.array(self.targets)

    @property
    def prompt(self):
        return ORIGINAL_PROMPT

    def _data_dict(self):
        train_dataset = TorchCIFAR100(self.root, train=True)
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset.data)):
            train_data_dict[train_dataset.targets[i]].append(str(train_dataset.data[i]))
        return train_data_dict

    @staticmethod
    def loader(data):
        return fromarray(data)


class CIFAR100Text(VLMDataset, Dataset):
    dataset_path = 'cifar-100-python'
    n_class = 100

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0, is_train=False):
        dataset = TorchCIFAR100(root, train=split == 'train')
        class_name_list = CIFAR100_CLASS_NAME
        super().__init__(root, dataset.data, dataset.targets, class_name_list, transform, target_transform, n_shot)
        self.targets = np.array(self.targets)
        self.augmentation_prompt = AUGMENT_PROMPT
        if is_train:
            self.__getitem_fn = self.__getitem_train

    @property
    def prompt(self):
        return ORIGINAL_PROMPT

    def _data_dict(self):
        train_dataset = TorchCIFAR100(self.root, train=True)
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset.data)):
            train_data_dict[train_dataset.targets[i]].append(str(train_dataset.data[i]))
        return train_data_dict

    @staticmethod
    def loader(data):
        return fromarray(data)

    def setup_prompt_transform(self):
        self.pre_processing = transforms.Compose(self.transform.transforms[:2])
        self.randaug = RandAugment(**self.transform.transforms[2].__dict__)
        self.post_processing = transforms.Compose(self.transform.transforms[3:])

    def ra_prompt(self, ra_tf, target):
        prompt = random.choice(self.augmentation_prompt)
        ra_fs = ''
        ra_fs += f'{RAND_AUG_TRANSFORMS[ra_tf[0].name]} and '
        ra_fs += f'{RAND_AUG_TRANSFORMS[ra_tf[1].name]}'

        prompt = prompt(ra_fs, self.num2str(target))
        return prompt

    def org_prompt(self, target):
        prompt = random.choice(self.augmentation_prompt)
        prompt = prompt('original', self.num2str(target))
        return prompt

    def __getitem__train(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)

        imgs = self.pre_processing(imgs)
        ra_imgs, ra_tf = self.randaug(imgs)
        ra_imgs = self.post_processing(ra_imgs)
        imgs = self.post_processing(imgs)

        return imgs, ra_imgs, target, self.org_prompt(target), self.ra_prompt(ra_tf, target)

    def __getitem_train(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)

        imgs = self.pre_processing(imgs)
        ra_imgs, ra_tf = self.randaug(imgs)
        ra_imgs = self.post_processing(ra_imgs)
        imgs = self.post_processing(imgs)

        return imgs, ra_imgs, target, self.org_prompt(target), self.ra_prompt(ra_tf, target)

    def __getitem_fn(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)
        imgs = self.transform(imgs)
        return imgs, target

    def __getitem__(self, idx):
        return self.__getitem_fn(idx)



if __name__ == '__main__':
    ds = CIFAR100('/data', transform=transforms.ToTensor())

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("Abyssinian")}, {ds.num2str(data[1])}')
