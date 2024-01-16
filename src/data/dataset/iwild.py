import os
import random

import numpy as np
import wilds
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.data.dataset import VLMDataset, IWILDCAM_CLASS_NAME
from .ra_text import RandAugment, RAND_AUG_TRANSFORMS

ORIGINAL_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'{name} in the wild.',
]

AUGMENT_PROMPT = [
    lambda augment, name: f'a {augment} photo of a {name}.',
    lambda augment, name: f'{augment} {name} in the wild.',
]


class IWildCam(VLMDataset, Dataset):
    dataset_path = 'iwildcam_v2.0'
    n_class = 182

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        assert split in ['train', 'val', 'test', 'id_val', 'id_test']
        self.dataset = wilds.get_dataset(dataset='iwildcam', root_dir=root)
        class_name_list = IWILDCAM_CLASS_NAME
        super().__init__(root, *self._imgs_targets(self.dataset, split, root), class_name_list, transform, target_transform,
                         n_shot)

    def _imgs_targets(self, dataset, split, root):
        split_mask = dataset.split_array == dataset.split_dict[split]
        split_idx = np.where(split_mask)[0]

        imgs = dataset._input_array[split_idx]
        for i in range(len(imgs)):
            imgs[i] = os.path.join(root, self.dataset_path, 'train', imgs[i])
        targets = dataset.y_array[split_idx].numpy()
        self.meta_data = dataset.metadata_array

        return imgs, targets

    def scoring(self, y_pred, y_true):
        return self.dataset.eval(y_pred, y_true, self.meta_data)

    @property
    def prompt(self):
        return ORIGINAL_PROMPT


class IWildCamText(VLMDataset, Dataset):
    dataset_path = 'iwildcam_v2.0'
    n_class = 182

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        assert split in ['train', 'val', 'test', 'id_val', 'id_test']
        dataset = wilds.get_dataset(dataset='iwildcam', root_dir=root)
        class_name_list = IWILDCAM_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset, split, root), class_name_list, transform, target_transform,
                         n_shot)
        self.augmentation_prompt = AUGMENT_PROMPT

    def _imgs_targets(self, dataset, split, root):
        split_mask = dataset.split_array == dataset.split_dict[split]
        split_idx = np.where(split_mask)[0]

        imgs = dataset._input_array[split_idx]
        for i in range(len(imgs)):
            imgs[i] = os.path.join(root, self.dataset_path, 'train', imgs[i])
        targets = dataset._y_array[split_idx].numpy()

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
    import torchvision.transforms as tf

    ds = IWildCam(root='/data', split='id_test', transform=tf.ToTensor())
    print(len(ds))
    print(np.unique(ds.targets))
    print(ds.imgs[0])
