import json
import os.path
import random
from abc import ABC
from collections import defaultdict

from PIL import Image
from termcolor import colored


class VLMDataset(ABC):
    root: str
    dataset_path: str
    n_class: int
    class_name: list
    _num2name: dict
    _name2num: dict
    prompt: list
    imgs: list
    targets: list

    def __init__(self, root, imgs, targets, class_name_list, transform, target_transform, n_shot):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.n_shot = n_shot

        self.set_class_name(class_name_list)
        self.prompt = self.set_prompt()
        self.origin_imgs, self.origin_targets = imgs, targets
        self.sampling(n_shot)

    @staticmethod
    def set_prompt():
        """
        list of prompts
        """
        prompt = ["a photo of {}."]
        return prompt

    def set_class_name(self, class_name_list):
        self.class_name = class_name_list
        self._num2name = {i: name for i, name in enumerate(class_name_list)}
        self._name2num = {name: i for i, name in enumerate(class_name_list)}

    def num2str(self, num):
        return self._num2name.get(num, f'{num} is not exists in {self.dataset_path}')

    def str2num(self, class_name):
        return self._name2num.get(class_name, f'{class_name} is not exists in {self.dataset_path}')

    def _data_dict(self):
        data_dict = defaultdict(list)
        for i in range(len(self.origin_imgs)):
            data_dict[self.origin_targets[i]].append(self.origin_imgs[i])
        return data_dict

    def sampling(self, n_shot):
        self.n_shot = n_shot

        if n_shot == 0:
            self.imgs, self.targets = self.origin_imgs, self.origin_targets
            return

        sample_path = os.path.join(self.root, 'sampling', f'soonge_{self.dataset_path}_{n_shot}s.json')

        if os.path.exists(sample_path):
            print(f'Load exist samples: {sample_path}')
            with open(sample_path, 'r') as f:
                data = json.load(f)
                self.imgs, self.targets = data['s_imgs'], data['s_targets']
                return

        # Generate N_way K_shot samples.
        s_imgs = list()
        s_targets = list()
        for class_num, items in self._data_dict().items():
            s_imgs.extend(random.sample(items, n_shot))
            s_targets.extend([class_num for _ in range(n_shot)])

        with open(sample_path, 'w') as f:
            print(f'Generate new samples: {sample_path}')
            json.dump(dict(s_imgs=s_imgs, s_targets=s_targets), f, indent=4, ensure_ascii=False)

        self.imgs, self.targets = s_imgs, s_targets

    @staticmethod
    def loader(path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, target = self.imgs[idx], self.targets[idx]
        imgs = self.loader(path)
        if self.transform is not None:
            imgs = self.transform(imgs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return imgs, target

    def __str__(self):
        return f'{self.__class__.__name__} | # class: {self.n_class} | # root: {self.dataset_path} | prompt: {self.prompt}'

    @staticmethod
    def _split_warning(dataset_name, split, state):
        if split is not state:
            print(f'{colored("[DATASET_WARNING]", "red")} {dataset_name} is not supported "split" argument.')