import os
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet as TorchImagenet
from torchvision.transforms import transforms

from . import VLMDataset, IMAGENET_CLASS_NAME


class ImageNet(VLMDataset, Dataset):
    dataset_path = 'imageNet'
    n_class = 1000

    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        dataset = TorchImagenet(os.path.join(root, self.dataset_path), split)
        class_name_list = IMAGENET_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return [
            lambda c: f'itap of a {c}.',
            lambda c: f'a bad photo of the {c}.',
            lambda c: f'a origami {c}.',
            lambda c: f'a photo of the large {c}.',
            lambda c: f'a {c} in a video game.',
            lambda c: f'art of the {c}.',
            lambda c: f'a photo of the small {c}.',
        ]

    def _data_dict(self):
        train_dataset = TorchImagenet(os.path.join(self.root, self.dataset_path), 'train')
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset.imgs)):
            train_data_dict[train_dataset.targets[i]].append(train_dataset.imgs[i][0])

        return train_data_dict


RAND_AUG_TRANSFORMS = {
    'AutoContrast': 'auto contrasted',
    'Equalize': 'equalized',
    'Invert': 'inverted',
    'Rotate': 'rotated',
    'Posterize': 'posterized',
    'Solarize': 'solarized',
    'Color': 'colored',
    'Contrast': 'contrasted',
    'Brightness': 'brighter',
    'BrightnessIncreasing': 'more brighter',
    'Sharpness': 'sharper',
    'PosterizeIncreasing': 'more posterized',
    'SolarizeAdd': 'adding solarized',
    'SolarizeIncreasing': 'increasing solarized',
    'ColorIncreasing': 'color factor increased',
    'ContrastIncreasing': 'contrasted',
    'SharpnessIncreasing': 'more sharper',
    'ShearX': 'shear to x',
    'ShearY': 'shear to y',
    'TranslateXRel': 'translated by x',
    'TranslateYRel': 'translated by y',
}


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights
        self.replace = self.choice_weights is None

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.replace,
            p=self.choice_weights,
        )
        for op in ops:
            img = op(img)
        return img, ops

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.num_layers}, ops='
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += ')'
        return fs


class ImageNetRandaugPrompt(ImageNet):
    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        super().__init__(root, split, transform, target_transform, n_shot)

        self.augmentation_prompt = [
            lambda augment, name: f'{augment} itap of a {name}.',
            lambda augment, name: f'itap of a {augment} {name}.',
            lambda augment, name: f'a bad {augment} photo of the {name}.',
            lambda augment, name: f'a {augment} origami {name}.',
            lambda augment, name: f'a {augment} {name} in a video game.',
            lambda augment, name: f'{augment} art of the {name}.',
            lambda augment, name: f'art of the {augment} {name}.',
            lambda augment, name: f'a {augment} photo of the {name}.',
            lambda augment, name: f'{augment} transformed image of {name}.',
            lambda augment, name: f'{augment} transformed photo of the {name}.',
        ]
        self.len_prompt = len(self.augmentation_prompt)

    def setup_prompt_transform(self):
        self.pre_processing = transforms.Compose(self.transform.transforms[:2])
        self.randaug = RandAugment(**self.transform.transforms[2].__dict__)
        self.post_processing = transforms.Compose(self.transform.transforms[3:])

    def ra_prompt(self, idx, ra_tf, target):
        prompt = self.augmentation_prompt[idx % self.len_prompt]
        ra_fs = ''
        ra_fs += f'{RAND_AUG_TRANSFORMS[ra_tf[0].name]} and '
        ra_fs += f'{RAND_AUG_TRANSFORMS[ra_tf[1].name]}'

        prompt = prompt(ra_fs, self.num2str(target))
        return prompt

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)
        if self.transform is not None:
            imgs = self.pre_processing(imgs)
            imgs, ra_tf = self.randaug(imgs)
            imgs = self.post_processing(imgs)

        return imgs, target, self.ra_prompt(idx, ra_tf, target)


class ImageNetRandaugPromptV2(ImageNet):
    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        super().__init__(root, split, transform, target_transform, n_shot)

        self.augmentation_prompt = [
            lambda augment, name: f'{augment} itap of a {name}.',
            lambda augment, name: f'itap of a {augment} {name}.',
            lambda augment, name: f'a bad {augment} photo of the {name}.',
            lambda augment, name: f'a {augment} origami {name}.',
            lambda augment, name: f'a {augment} {name} in a video game.',
            lambda augment, name: f'{augment} art of the {name}.',
            lambda augment, name: f'art of the {augment} {name}.',
            lambda augment, name: f'a {augment} photo of the {name}.',
            lambda augment, name: f'{augment} transformed image of {name}.',
            lambda augment, name: f'{augment} transformed photo of the {name}.',
        ]
        self.len_prompt = len(self.augmentation_prompt)

    def setup_prompt_transform(self):
        self.pre_processing = transforms.Compose(self.transform.transforms[:2])
        self.randaug = RandAugment(**self.transform.transforms[2].__dict__)
        self.post_processing = transforms.Compose(self.transform.transforms[3:])

    def ra_prompt(self, idx, ra_tf, target):
        prompt = self.augmentation_prompt[idx % self.len_prompt]
        ra_fs = ''
        ra_fs += f'{RAND_AUG_TRANSFORMS[ra_tf[0].name]} and '
        ra_fs += f'{RAND_AUG_TRANSFORMS[ra_tf[1].name]}'

        prompt = prompt(ra_fs, self.num2str(target))
        return prompt

    def original_prompt(self, idx, target):
        prompt = self.augmentation_prompt[idx % self.len_prompt]
        return prompt('original', self.num2str(target))

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.pre_processing(img)
            ra_img, ra_tf = self.randaug(img)
            ra_img = self.post_processing(ra_img)
            img = self.post_processing(img)

        return img, ra_img, target, self.original_prompt(idx, target), self.ra_prompt(idx, ra_tf, target)

AUGPROMPT = [
            lambda augment, name: f'{augment} itap of a {name}.',
            lambda augment, name: f'itap of a {augment} {name}.',
            lambda augment, name: f'a bad {augment} photo of the {name}.',
            lambda augment, name: f'a {augment} origami {name}.',
            lambda augment, name: f'a {augment} {name} in a video game.',
            lambda augment, name: f'{augment} art of the {name}.',
            lambda augment, name: f'art of the {augment} {name}.',
            lambda augment, name: f'a {augment} photo of the {name}.',
            lambda augment, name: f'{augment} transformed image of {name}.',
            lambda augment, name: f'{augment} transformed photo of the {name}.',
        ]

if __name__ == '__main__':
    ds = ImageNet('/data', transform=transforms.ToTensor(), n_shot=0)
    ds.sampling(1)
    print(len(ds))
    data = next(iter(ds))
    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("tench")}, {ds.num2str(data[1])}')
