import os
import random
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

AUGMENT_PROMPT = [
    lambda augment, name: f'{augment} itap of a {name}.',
    lambda augment, name: f'a {augment} bad photo of a {name}.',
    lambda augment, name: f'a {augment} photo of many {name}.',
    lambda augment, name: f'a {augment} sculpture of a {name}.',
    lambda augment, name: f'a {augment} photo of the hard to see {name}.',
    lambda augment, name: f'a {augment} low resolution photo of the {name}.',
    lambda augment, name: f'a {augment} rendering of a {name}.',
    lambda augment, name: f'{augment} graffiti of a {name}.',
    lambda augment, name: f'a {augment} bad photo of the {name}.',
    lambda augment, name: f'a {augment} cropped photo of the {name}.',
    lambda augment, name: f'a {augment} tattoo of a {name}.',
    lambda augment, name: f'the embroidered {augment} {name}.',
    lambda augment, name: f'a {augment} photo of a hard to see {name}.',
    lambda augment, name: f'a {augment} photo of a {name}.',  # brighten
    lambda augment, name: f'a {augment} photo of a clean {name}.',
    lambda augment, name: f'a {augment} photo of a dirty {name}.',
    lambda augment, name: f'a {augment} dark photo of the {name}.',
    lambda augment, name: f'a {augment} drawing of a {name}.',
    lambda augment, name: f'a {augment} photo of my {name}.',
    lambda augment, name: f'the {augment} plastic {name}.',
    lambda augment, name: f'a {augment} photo of the cool {name}.',
    lambda augment, name: f'a close-up {augment} photo of a {name}.',
    lambda augment, name: f'a black and white {augment} photo of the {name}.',
    lambda augment, name: f'a {augment} painting of the {name}.',
    lambda augment, name: f'a {augment} painting of a {name}.',
    lambda augment, name: f'a {augment} pixelated photo of the {name}.',
    lambda augment, name: f'a {augment} sculpture of the {name}.',
    lambda augment, name: f'a {augment} bright photo of the {name}.',
    lambda augment, name: f'a {augment} cropped photo of a {name}.',
    lambda augment, name: f'a {augment} plastic {name}.',
    lambda augment, name: f'a {augment} photo of the dirty {name}.',
    lambda augment, name: f'a jpeg corrupted {augment} photo of a {name}.',
    lambda augment, name: f'a blurry {augment} photo of the {name}.',
    lambda augment, name: f'a {augment} photo of the {name}.',
    lambda augment, name: f'a good {augment} photo of the {name}.',
    lambda augment, name: f'a {augment} rendering of the {name}.',
    lambda augment, name: f'a {name} in a {augment}  video game.',
    lambda augment, name: f'a {augment} photo of one {name}.',
    lambda augment, name: f'a {augment} doodle of a {name}.',
    lambda augment, name: f'a {augment} close-up photo of the {name}.',
    lambda augment, name: f'a {augment} photo of a {name}.',
    lambda augment, name: f'the {augment} origami {name}.',
    lambda augment, name: f'the {name} in a {augment} video game.',
    lambda augment, name: f'a {augment} sketch of a {name}.',
    lambda augment, name: f'a {augment} doodle of the {name}.',
    lambda augment, name: f'a {augment} origami {name}.',
    lambda augment, name: f'a {augment} low resolution photo of a {name}.',
    lambda augment, name: f'the {augment} toy {name}.',
    lambda augment, name: f'a {augment} rendition of the {name}.',
    lambda augment, name: f'a {augment} photo of the clean {name}.',
    lambda augment, name: f'a {augment} photo of a large {name}.',
    lambda augment, name: f'a {augment} rendition of a {name}.',
    lambda augment, name: f'a {augment} photo of a nice {name}.',
    lambda augment, name: f'a {augment} photo of a weird {name}.',
    lambda augment, name: f'a {augment} blurry photo of a {name}.',
    lambda augment, name: f'a {augment} cartoon {name}.',
    lambda augment, name: f'{augment} art of a {name}.',
    lambda augment, name: f'a {augment} sketch of the {name}.',
    lambda augment, name: f'a {augment} embroidered {name}.',
    lambda augment, name: f'a {augment} pixelated photo of a {name}.',
    lambda augment, name: f'{augment} itap of the {name}.',
    lambda augment, name: f'a jpeg corrupted {augment} photo of the {name}.',
    lambda augment, name: f'a good {augment} photo of a {name}.',
    lambda augment, name: f'a {augment} plushie {name}.',
    lambda augment, name: f'a {augment} photo of the nice {name}.',
    lambda augment, name: f'a {augment} photo of the small {name}.',
    lambda augment, name: f'a {augment} photo of the weird {name}.',
    lambda augment, name: f'the {augment} cartoon {name}.',
    lambda augment, name: f'{augment} art of the {name}.',
    lambda augment, name: f'a {augment} drawing of the {name}.',
    lambda augment, name: f'a {augment} photo of the large {name}.',
    lambda augment, name: f'a black and white {augment} photo of a {name}.',
    lambda augment, name: f'the {augment} plushie {name}.',
    lambda augment, name: f'a dark {augment} photo of a {name}.',
    lambda augment, name: f'{augment} itap of a {name}.',
    lambda augment, name: f'{augment} graffiti of the {name}.',
    lambda augment, name: f'a {augment} toy {name}.',
    lambda augment, name: f'{augment} itap of my {name}.',
    lambda augment, name: f'a {augment} photo of a cool {name}.',
    lambda augment, name: f'a {augment} photo of a small {name}.',
    lambda augment, name: f'a {augment} tattoo of the {name}.',
]
ORIGINAL_PROMPT = [
    lambda name: f'a bad photo of a {name}.',
    lambda name: f'a photo of many {name}.',
    lambda name: f'a sculpture of a {name}.',
    lambda name: f'a photo of the hard to see {name}.',
    lambda name: f'a low resolution photo of the {name}.',
    lambda name: f'a rendering of a {name}.',
    lambda name: f'graffiti of a {name}.',
    lambda name: f'a bad photo of the {name}.',
    lambda name: f'a cropped photo of the {name}.',
    lambda name: f'a tattoo of a {name}.',
    lambda name: f'the embroidered {name}.',
    lambda name: f'a photo of a hard to see {name}.',
    lambda name: f'a bright photo of a {name}.',
    lambda name: f'a photo of a clean {name}.',
    lambda name: f'a photo of a dirty {name}.',
    lambda name: f'a dark photo of the {name}.',
    lambda name: f'a drawing of a {name}.',
    lambda name: f'a photo of my {name}.',
    lambda name: f'the plastic {name}.',
    lambda name: f'a photo of the cool {name}.',
    lambda name: f'a close-up photo of a {name}.',
    lambda name: f'a black and white photo of the {name}.',
    lambda name: f'a painting of the {name}.',
    lambda name: f'a painting of a {name}.',
    lambda name: f'a pixelated photo of the {name}.',
    lambda name: f'a sculpture of the {name}.',
    lambda name: f'a bright photo of the {name}.',
    lambda name: f'a cropped photo of a {name}.',
    lambda name: f'a plastic {name}.',
    lambda name: f'a photo of the dirty {name}.',
    lambda name: f'a jpeg corrupted photo of a {name}.',
    lambda name: f'a blurry photo of the {name}.',
    lambda name: f'a photo of the {name}.',
    lambda name: f'a good photo of the {name}.',
    lambda name: f'a rendering of the {name}.',
    lambda name: f'a {name} in a video game.',
    lambda name: f'a photo of one {name}.',
    lambda name: f'a doodle of a {name}.',
    lambda name: f'a close-up photo of the {name}.',
    lambda name: f'a photo of a {name}.',
    lambda name: f'the origami {name}.',
    lambda name: f'the {name} in a video game.',
    lambda name: f'a sketch of a {name}.',
    lambda name: f'a doodle of the {name}.',
    lambda name: f'a origami {name}.',
    lambda name: f'a low resolution photo of a {name}.',
    lambda name: f'the toy {name}.',
    lambda name: f'a rendition of the {name}.',
    lambda name: f'a photo of the clean {name}.',
    lambda name: f'a photo of a large {name}.',
    lambda name: f'a rendition of a {name}.',
    lambda name: f'a photo of a nice {name}.',
    lambda name: f'a photo of a weird {name}.',
    lambda name: f'a blurry photo of a {name}.',
    lambda name: f'a cartoon {name}.',
    lambda name: f'art of a {name}.',
    lambda name: f'a sketch of the {name}.',
    lambda name: f'a embroidered {name}.',
    lambda name: f'a pixelated photo of a {name}.',
    lambda name: f'itap of the {name}.',
    lambda name: f'a jpeg corrupted photo of the {name}.',
    lambda name: f'a good photo of a {name}.',
    lambda name: f'a plushie {name}.',
    lambda name: f'a photo of the nice {name}.',
    lambda name: f'a photo of the small {name}.',
    lambda name: f'a photo of the weird {name}.',
    lambda name: f'the cartoon {name}.',
    lambda name: f'art of the {name}.',
    lambda name: f'a drawing of the {name}.',
    lambda name: f'a photo of the large {name}.',
    lambda name: f'a black and white photo of a {name}.',
    lambda name: f'the plushie {name}.',
    lambda name: f'a dark photo of a {name}.',
    lambda name: f'itap of a {name}.',
    lambda name: f'graffiti of the {name}.',
    lambda name: f'a toy {name}.',
    lambda name: f'itap of my {name}.',
    lambda name: f'a photo of a cool {name}.',
    lambda name: f'a photo of a small {name}.',
    lambda name: f'a tattoo of the {name}.',
]


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


class ImageNetRandaugPromptText(ImageNet):
    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        super().__init__(root, split, transform, target_transform, n_shot)
        self.augmentation_prompt = AUGMENT_PROMPT
        self.original_prompt = ORIGINAL_PROMPT

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
        prompt = random.choice(self.original_prompt)
        prompt = prompt(self.num2str(target))
        return prompt

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)

        imgs = self.pre_processing(imgs)
        ra_imgs, ra_tf = self.randaug(imgs)
        ra_imgs = self.post_processing(ra_imgs)
        imgs = self.post_processing(imgs)

        return imgs, ra_imgs, target, self.org_prompt(target), self.ra_prompt(ra_tf, target)


class ImageNetRandaugPromptOriginalText(ImageNet):
    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0, is_train=False):
        super().__init__(root, split, transform, target_transform, n_shot)
        self.augmentation_prompt = AUGMENT_PROMPT
        self.original_prompt = ORIGINAL_PROMPT

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

    def sim_prompt(self, target):
        prompt = random.choice(ORIGINAL_PROMPT)
        prompt = prompt(self.num2str(target))
        return prompt

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)

        imgs = self.pre_processing(imgs)
        ra_imgs, ra_tf = self.randaug(imgs)
        ra_imgs = self.post_processing(ra_imgs)
        imgs = self.post_processing(imgs)

        return imgs, ra_imgs, target, self.org_prompt(target), self.ra_prompt(ra_tf, target)


class ImageNetSimplePromptText(ImageNet):
    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0, is_train=False):
        super().__init__(root, split, transform, target_transform, n_shot)
        self.original_prompt = ORIGINAL_PROMPT

    def setup_prompt_transform(self):
        pass

    def org_prompt(self, target):
        prompt = random.choice(self.original_prompt)
        prompt = prompt(self.num2str(target))
        return prompt

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)
        imgs = self.transform(imgs)

        return imgs, target, self.org_prompt(target)


class ImageNetSimpleAugPrompt(ImageNetRandaugPromptOriginalText):
    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)

        imgs = self.pre_processing(imgs)
        ra_imgs, ra_tf = self.randaug(imgs)
        ra_imgs = self.post_processing(ra_imgs)

        return ra_imgs, target, self.ra_prompt(ra_tf, target)


class ImageNetSimpleAugNormPrompt(ImageNetRandaugPromptOriginalText):
    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)

        imgs = self.pre_processing(imgs)
        ra_imgs, ra_tf = self.randaug(imgs)
        ra_imgs = self.post_processing(ra_imgs)

        return ra_imgs, target, self.org_prompt(target)


class OriginalTextImageNetRandaugPromptAblationNN(ImageNetRandaugPromptOriginalText):
    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        img = self.loader(path)

        img1 = self.pre_processing(img)
        img2 = self.pre_processing(img)

        img1 = self.post_processing(img1)
        img2 = self.post_processing(img2)

        return img1, img2, target, self.sim_prompt(target), self.sim_prompt(target)


class OriginalTextImageNetRandaugPromptAblationAN(ImageNetRandaugPromptOriginalText):
    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)

        imgs = self.pre_processing(imgs)
        ra_imgs, ra_tf = self.randaug(imgs)
        ra_imgs = self.post_processing(ra_imgs)
        imgs = self.post_processing(imgs)

        return imgs, ra_imgs, target, self.sim_prompt(target), self.sim_prompt(target)


class OriginalTextImageNetRandaugPromptAblationAA(ImageNetRandaugPromptOriginalText):
    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)

        imgs = self.pre_processing(imgs)
        ra_imgs, ra_tf = self.randaug(imgs)
        ra_imgs = self.post_processing(ra_imgs)

        imgs2 = self.pre_processing(imgs)
        ra_imgs2, ra_tf2 = self.randaug(imgs2)
        ra_imgs2 = self.post_processing(ra_imgs2)

        return ra_imgs, ra_imgs2, target, self.ra_prompt(ra_tf, target), self.ra_prompt(ra_tf2, target)


if __name__ == '__main__':
    ds = ImageNet('/data', transform=transforms.ToTensor(), n_shot=0)
    ds.sampling(1)
    print(len(ds))
    data = next(iter(ds))
    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("tench")}, {ds.num2str(data[1])}')
