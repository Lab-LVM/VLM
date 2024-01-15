import random
from collections import defaultdict

from PIL.Image import fromarray
from torch.utils.data import Dataset
from torchvision.datasets import PCAM as TorchPCAM
from torchvision.transforms import transforms

from . import PCAM_CLASS_NAME
from . import VLMDataset
from .ra_text import RandAugment, RAND_AUG_TRANSFORMS

ORIGINAL_PROMPT = [
    lambda name: f'this is a photo of {name}',
    # lambda name: f'a histopathology slide showing {name}',
    # lambda name: f'histopathology image of {name}',
]

AUGMENT_PROMPT = [
    lambda augment, name: f'this is a {augment} photo of {name}',
    # lambda augment, name: f'a histopathology {augment} slide showing {name}',
    # lambda augment, name: f'histopathology {augment} image of {name}',
]


class PCam(VLMDataset, Dataset):
    dataset_path = 'pcam'
    n_class = 2

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchPCAM(root, split)
        img = dataset.h5py.File(dataset._base_folder / dataset._FILES[dataset._split]['images'][0])['x'][:]
        target = dataset.h5py.File(dataset._base_folder / dataset._FILES[dataset._split]['targets'][0])['y'][:, 0, 0, 0]
        class_name_list = PCAM_CLASS_NAME
        super().__init__(root, img, target, class_name_list, transform, target_transform, n_shot)

    @property
    def prompt(self):
        return [
            lambda c: f'this is a photo of {c}',
            lambda c: f'a histopathology slide showing {c}',
            lambda c: f'histopathology image of {c}'
        ]

    def _data_dict(self):
        t_dataset = TorchPCAM(self.root, split='train')
        train_data_dict = defaultdict(list)

        img = t_dataset.h5py.File(t_dataset._base_folder / t_dataset._FILES[t_dataset._split]['images'][0])['x'][:]
        target = t_dataset.h5py.File(t_dataset._base_folder / t_dataset._FILES[t_dataset._split]['targets'][0])['y'][:,
                 0, 0, 0]
        for i in range(len(img)):
            train_data_dict[target[i]].append(str(img[i]))
        return train_data_dict

    @staticmethod
    def loader(data):
        return fromarray(data).convert("RGB")


class PCamraText(VLMDataset, Dataset):
    dataset_path = 'pcam'
    n_class = 2

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchPCAM(root, split)
        img = dataset.h5py.File(dataset._base_folder / dataset._FILES[dataset._split]['images'][0])['x'][:]
        target = dataset.h5py.File(dataset._base_folder / dataset._FILES[dataset._split]['targets'][0])['y'][:, 0, 0, 0]
        class_name_list = PCAM_CLASS_NAME
        super().__init__(root, img, target, class_name_list, transform, target_transform, n_shot)
        self.augmentation_prompt = AUGMENT_PROMPT

    @property
    def prompt(self):
        return ORIGINAL_PROMPT

    def _data_dict(self):
        t_dataset = TorchPCAM(self.root, split='train')
        train_data_dict = defaultdict(list)

        img = t_dataset.h5py.File(t_dataset._base_folder / t_dataset._FILES[t_dataset._split]['images'][0])['x'][:]
        target = t_dataset.h5py.File(t_dataset._base_folder / t_dataset._FILES[t_dataset._split]['targets'][0])['y'][:,
                 0, 0, 0]
        for i in range(len(img)):
            train_data_dict[target[i]].append(img[i])
        return train_data_dict

    @staticmethod
    def loader(data):
        return fromarray(data).convert("RGB")

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
    ds = PCam('/data', transform=transforms.ToTensor())

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name)
