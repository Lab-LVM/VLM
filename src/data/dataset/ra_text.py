import random

import numpy as np
from torchvision.transforms import transforms

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

class RATextWrapper:
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
