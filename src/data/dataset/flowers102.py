import random
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import Flowers102 as TorchFlowers102
from torchvision.transforms import transforms

from . import VLMDataset, FLOWERS102_CLASS_NAME
from .ra_text import RandAugment, RAND_AUG_TRANSFORMS

ORIGINAL_PROMPT = [
    lambda name: f'a photo of a {name}, a type of flower.'
]
AUGMENT_PROMPT = [
    lambda augment, name: f'a {augment} photo of a {name}, a type of flower.'
]


class Flowers102(VLMDataset, Dataset):
    dataset_path = 'flowers-102'
    n_class = 102

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        if split == 'trainval':
            train_dataset = TorchFlowers102(root, 'train')
            val_dataset = TorchFlowers102(root, 'val')
            train_dataset._image_files.extend(val_dataset._image_files)
            train_dataset._labels.extend(val_dataset._labels)
            dataset = train_dataset
        else:
            dataset = TorchFlowers102(root, split)
        class_name_list = FLOWERS102_CLASS_NAME
        super().__init__(root, dataset._image_files, dataset._labels, class_name_list, transform, target_transform,
                         n_shot)

    @property
    def prompt(self):
        return ORIGINAL_PROMPT

    def _data_dict(self):
        train_dataset = TorchFlowers102(self.root, 'train')
        val_dataset = TorchFlowers102(self.root, 'val')
        train_dataset._image_files.extend(val_dataset._image_files)
        train_dataset._labels.extend(val_dataset._labels)

        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset)):
            train_data_dict[train_dataset._labels[i]].append(str(train_dataset._image_files[i]))
        return train_data_dict


class Flowers102raText(VLMDataset, Dataset):
    dataset_path = 'flowers-102'
    n_class = 102

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        if split == 'trainval':
            train_dataset = TorchFlowers102(root, 'train')
            val_dataset = TorchFlowers102(root, 'val')
            train_dataset._image_files.extend(val_dataset._image_files)
            train_dataset._labels.extend(val_dataset._labels)
            dataset = train_dataset
        else:
            dataset = TorchFlowers102(root, split)
        class_name_list = FLOWERS102_CLASS_NAME
        super().__init__(root, dataset._image_files, dataset._labels, class_name_list, transform, target_transform,
                         n_shot)
        self.augmentation_prompt = AUGMENT_PROMPT

    @property
    def prompt(self):
        return ORIGINAL_PROMPT

    def _data_dict(self):
        train_dataset = TorchFlowers102(self.root, 'train')
        val_dataset = TorchFlowers102(self.root, 'val')
        train_dataset._image_files.extend(val_dataset._image_files)
        train_dataset._labels.extend(val_dataset._labels)

        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset)):
            train_data_dict[train_dataset._labels[i]].append(str(train_dataset._image_files[i]))
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
    ds = Flowers102('/data/vlm', transform=transforms.ToTensor(), n_shot=3)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("pink primrose")}, {ds.num2str(data[1])}')
