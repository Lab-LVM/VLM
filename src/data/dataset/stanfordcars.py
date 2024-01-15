import random
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import StanfordCars as TorchStanfordCars
from torchvision.transforms import transforms

from . import VLMDataset, STANFORDCARS_CLASS_NAME
from .ra_text import RandAugment, RAND_AUG_TRANSFORMS

ORIGINAL_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'a photo of the {name}.',
    lambda name: f'a photo of my {name}.',
    lambda name: f'i love my {name}!',
    lambda name: f'a photo of my dirty {name}.',
    lambda name: f'a photo of my clean {name}.',
    lambda name: f'a photo of my new {name}.',
    lambda name: f'a photo of my old {name}.',
]
AUGMENT_PROMPT = [
    lambda augment, name: f'a {augment} photo of a {name}.',
    lambda augment, name: f'a {augment} photo of the {name}.',
    lambda augment, name: f'a {augment} photo of my {name}.',
    lambda augment, name: f'i love my {augment} {name}!',
    lambda augment, name: f'a {augment} photo of my dirty {name}.',
    lambda augment, name: f'a {augment} photo of my clean {name}.',
    lambda augment, name: f'a {augment} photo of my new {name}.',
    lambda augment, name: f'a {augment} photo of my old {name}.',
]


class StanfordCars(VLMDataset, Dataset):
    dataset_path = 'stanford_cars'
    n_class = 196

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchStanfordCars(root, split)
        class_name_list = STANFORDCARS_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

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
        return ORIGINAL_PROMPT

    def _data_dict(self):
        train_dataset = TorchStanfordCars(self.root, 'train')
        train_data_dict = defaultdict(list)
        _imgs, _targets = self._imgs_targets(train_dataset)

        for i in range(len(_imgs)):
            train_data_dict[_targets[i]].append(str(_imgs[i]))
        return train_data_dict


class StanfordCarsraText(VLMDataset, Dataset):
    dataset_path = 'stanford_cars'
    n_class = 196

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchStanfordCars(root, split)
        class_name_list = STANFORDCARS_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)
        self.augmentation_prompt = AUGMENT_PROMPT

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
        return ORIGINAL_PROMPT

    def _data_dict(self):
        train_dataset = TorchStanfordCars(self.root, 'train')
        train_data_dict = defaultdict(list)
        _imgs, _targets = self._imgs_targets(train_dataset)

        for i in range(len(_imgs)):
            train_data_dict[_targets[i]].append(str(_imgs[i]))
        return train_data_dict

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

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)

        imgs = self.pre_processing(imgs)
        ra_imgs, ra_tf = self.randaug(imgs)
        ra_imgs = self.post_processing(ra_imgs)
        imgs = self.post_processing(imgs)

        return imgs, ra_imgs, target, self.org_prompt(target), self.ra_prompt(ra_tf, target)


if __name__ == '__main__':
    ds = StanfordCars('/data/vlm', transform=transforms.ToTensor(), n_shot=3)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("AM General Hummer SUV 2000")}, {ds.num2str(data[1])}')
