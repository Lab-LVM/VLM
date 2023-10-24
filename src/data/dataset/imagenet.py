import os
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet as TorchImagenet
from torchvision.transforms import transforms

from src.data.dataset import VLMDataset, IMAGENET_CLASS_NAME


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

    @staticmethod
    def set_prompt():
        prompt = ["itap of a {}.",
                  "a bad photo of the {}.",
                  "a origami {}.",
                  "a photo of the large {}.",
                  "a {} in a video game.",
                  "art of the {}.",
                  "a photo of the small {}.",
                  ]
        return prompt

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

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
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

        self.augmentation_prompt = ["{} itap of a {}.",
                                    "itap of a {} {}.",
                                    "a bad {} photo of the {}.",
                                    "a {} origami {}.",
                                    "a {} {} in a video game.",
                                    "{} art of the {}.",
                                    "art of the {} {}.",
                                    "a {} photo of the {}.",
                                    "{} transformed image of {}.",
                                    "{} transformed photo of the {}.",
                                    ]

    def setup_prompt_transform(self):
        self.pre_processing = transforms.Compose(self.transform.transforms[:2])
        self.randaug = RandAugment(**self.transform.transforms[2].__dict__)
        self.post_processing = transforms.Compose(self.transform.transforms[3:])

    def ra_prompt(self, ra_tf, target):
        prompt = random.choice(self.augmentation_prompt)
        ra_fs = ''
        ra_fs += f'{RAND_AUG_TRANSFORMS[ra_tf[0].name]} and '
        ra_fs += f'{RAND_AUG_TRANSFORMS[ra_tf[1].name]}'

        prompt = prompt.format(ra_fs, self.num2str(target))
        return prompt

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)
        if self.transform is not None:
            imgs = self.pre_processing(imgs)
            imgs, ra_tf = self.randaug(imgs)
            imgs = self.post_processing(imgs)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs, target, self.ra_prompt(ra_tf, target)


if __name__ == '__main__':
    ds = ImageNet('/data/vlm', transform=transforms.ToTensor(), n_shot=0)
    ds.sampling(1)
    print(len(ds))
    data = next(iter(ds))
    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("tench")}, {ds.num2str(data[1])}')
