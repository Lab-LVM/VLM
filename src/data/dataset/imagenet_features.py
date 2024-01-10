import os
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.data.dataset import VLMDataset, IMAGENET_CLASS_NAME


class ImageNetRandaugPromptFeatures(VLMDataset):
    def __init__(self, root, backbone, split='train', transform=None, target_transform=None, n_shot=0,
                 dataset_path='imageNet_train_with_normal'):
        assert split == 'train', f'{self.__class__.name} only supports train split. Now is {split}.'
        super().__init__(root, None, None, IMAGENET_CLASS_NAME, None, None, 0)
        self.dataset_path = dataset_path
        self.backbone = backbone
        self.set_feature(0)

    def set_feature(self, index):
        file_name = os.path.join(self.root, f'{self.backbone}_{self.dataset_path}', f'{index}.pkl')
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        self.imgs = data['vision_features']
        self.prompts = data['language_features']
        self.imgs_aug = data['vision_features_aug']
        self.prompts_aug = data['language_features_aug']
        self.targets = data['targets']

    def __getitem__(self, idx):
        return self.imgs[idx], self.imgs_aug[idx], self.targets[idx], self.prompts[idx], self.prompts_aug[idx]


class ImageNetEvalFeatures(Dataset):
    def __init__(self, root, backbone, dataset_path='imagenet_ds_eval', dataset_name='imagenet', **kwargs):
        self.name = dataset_name
        pickle_file = os.path.join(root, f'{backbone}_{dataset_path}', f'{dataset_name}.pkl')

        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        self.imgs = data['vision_features']
        self.text = data['language_features']
        self.targets = data['targets']

        # language_type = kwargs.get('language_type', None)
        #
        # if language_type == 'original':
        #     self.text = data['origin_language_features']
        # elif language_type == 'modified':
        #     self.text = data['modified_language_features']
        # elif language_type == 'distorted':
        #     self.text = data['distorted_language_features']
        # elif language_type == 'ood': # out of distribution
        #     self.text = data['ood_language_features']
        # elif language_type == 'dko': # different kinds of
        #     self.text = data['dko_language_features']
        # elif language_type == 'na+ori': # natural+origin
        #     self.text = torch.cat([data['origin_language_features'], data['language_features']], dim=1)
        # elif language_type == 'ood+':
        #     self.text = torch.cat([data['ood_language_features'], data['dko_language_features']], dim=1)

    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]

    def __len__(self):
        return len(self.imgs)
