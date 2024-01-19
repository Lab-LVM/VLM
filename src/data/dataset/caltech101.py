import os
import random

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from src.data.dataset import VLMDataset, CALTECH101_CLASS_NAME
from src.data.dataset.ra_text import RandAugment, RAND_AUG_TRANSFORMS

ORIGINAL_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'a painting of a {name}.',
    lambda name: f'a plastic {name}.',
    lambda name: f'a sculpture of a {name}.',
    lambda name: f'a sketch of a {name}.',
    lambda name: f'a tattoo of a {name}.',
    lambda name: f'a toy {name}.',
    lambda name: f'a rendition of a {name}.',
    lambda name: f'a embroidered {name}.',
    lambda name: f'a cartoon {name}.',
    lambda name: f'a {name} in a video game.',
    lambda name: f'a plushie {name}.',
    lambda name: f'a origami {name}.',
    lambda name: f'art of a {name}.',
    lambda name: f'graffiti of a {name}.',
    lambda name: f'a drawing of a {name}.',
    lambda name: f'a doodle of a {name}.',
    lambda name: f'a photo of the {name}.',
    lambda name: f'a painting of the {name}.',
    lambda name: f'the plastic {name}.',
    lambda name: f'a sculpture of the {name}.',
    lambda name: f'a sketch of the {name}.',
    lambda name: f'a tattoo of the {name}.',
    lambda name: f'the toy {name}.',
    lambda name: f'a rendition of the {name}.',
    lambda name: f'the embroidered {name}.',
    lambda name: f'the cartoon {name}.',
    lambda name: f'the {name} in a video game.',
    lambda name: f'the plushie {name}.',
    lambda name: f'the origami {name}.',
    lambda name: f'art of the {name}.',
    lambda name: f'graffiti of the {name}.',
    lambda name: f'a drawing of the {name}.',
    lambda name: f'a doodle of the {name}.',
]

AUGMENT_PROMPT = [
    lambda augment, name: f'a {augment} photo of a {name}.',
    lambda augment, name: f'a {augment} painting of a {name}.',
    lambda augment, name: f'a {augment} plastic {name}.',
    lambda augment, name: f'a {augment} sculpture of a {name}.',
    lambda augment, name: f'a {augment} sketch of a {name}.',
    lambda augment, name: f'a {augment} tattoo of a {name}.',
    lambda augment, name: f'a {augment} toy {name}.',
    lambda augment, name: f'a {augment} rendition of a {name}.',
    lambda augment, name: f'a {augment} embroidered {name}.',
    lambda augment, name: f'a {augment} cartoon {name}.',
    lambda augment, name: f'a {name} in a {augment} video game.',
    lambda augment, name: f'a {augment} plushie {name}.',
    lambda augment, name: f'a {augment} origami {name}.',
    lambda augment, name: f'{augment} art of a {name}.',
    lambda augment, name: f'{augment} graffiti of a {name}.',
    lambda augment, name: f'a {augment} drawing of a {name}.',
    lambda augment, name: f'a {augment} doodle of a {name}.',
    lambda augment, name: f'a {augment} photo of the {name}.',
    lambda augment, name: f'a {augment} painting of the {name}.',
    lambda augment, name: f'the {augment} plastic {name}.',
    lambda augment, name: f'a {augment} sculpture of the {name}.',
    lambda augment, name: f'a {augment} sketch of the {name}.',
    lambda augment, name: f'a {augment} tattoo of the {name}.',
    lambda augment, name: f'the {augment} toy {name}.',
    lambda augment, name: f'a {augment} rendition of the {name}.',
    lambda augment, name: f'the {augment} embroidered {name}.',
    lambda augment, name: f'the {augment} cartoon {name}.',
    lambda augment, name: f'the {name} in a {augment} video game.',
    lambda augment, name: f'the {augment} plushie {name}.',
    lambda augment, name: f'the {augment} origami {name}.',
    lambda augment, name: f'{augment} art of the {name}.',
    lambda augment, name: f'{augment} graffiti of the {name}.',
    lambda augment, name: f'a {augment} drawing of the {name}.',
    lambda augment, name: f'a {augment} doodle of the {name}.',
]


class Caltech101(VLMDataset, Dataset):
    dataset_path = 'caltech101'
    n_class = 101

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = ImageFolder(os.path.join(root, self.dataset_path, split))
        class_name_list = CALTECH101_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return ORIGINAL_PROMPT


class Caltech101raText(VLMDataset, Dataset):
    dataset_path = 'caltech101'
    n_class = 101

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = ImageFolder(os.path.join(root, self.dataset_path, split))
        class_name_list = CALTECH101_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)
        self.augmentation_prompt = AUGMENT_PROMPT

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return ORIGINAL_PROMPT

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
    ds = Caltech101raText('/data', transform=transforms.ToTensor(), n_shot=4)

    data = next(iter(ds))
    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("background")}, {ds.num2str(data[1])}')