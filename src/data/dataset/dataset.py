import math
import os.path
import os.path
import pickle
import random
from abc import ABC
from collections import defaultdict

from PIL import Image
from termcolor import colored
from torchvision import transforms


def transform_eval(
        img_size=224,
        crop_pct=1.0,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),

):
    scale_size = math.floor(img_size / crop_pct)
    tfl = [
        transforms.Resize(scale_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    return transforms.Compose(tfl)


class VLMClassificationDataset(ABC):
    root: str
    dataset_path: str
    n_class: int
    class_names: list
    _num2name: dict
    _name2num: dict
    prompt: list
    imgs: list
    targets: list

    def __init__(self, root, imgs, targets, class_names, transform, target_transform, n_shot):
        self.root = root
        self.transform = transform if transform is not None else transform_eval()
        self.target_transform = target_transform
        self.n_shot = n_shot

        self.set_class_name(class_names)
        self.origin_imgs, self.origin_targets = imgs, targets
        self.sampling(n_shot)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def prompt(self):
        return [
            lambda c: f'a photo {c}.'
        ]

    def set_class_name(self, class_names):
        self.class_names = class_names
        self._num2name = {i: name for i, name in enumerate(class_names)}
        self._name2num = {name: i for i, name in enumerate(class_names)}

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

        sample_path = os.path.join(self.root, 'sampling', f'soonge_{self.dataset_path}_{n_shot}s.pkl')

        if os.path.exists(sample_path):
            print(f'Load exist samples: {sample_path}')
            with open(sample_path, 'rb') as f:
                data = pickle.load(f)
                self.imgs, self.targets = data['s_imgs'], data['s_targets']
                return

        # Generate N_way K_shot samples.
        s_imgs = list()
        s_targets = list()
        for class_num, items in self._data_dict().items():
            s_imgs.extend(random.sample(items, n_shot))
            s_targets.extend([class_num for _ in range(n_shot)])

        with open(sample_path, 'wb') as f:
            print(f'Generate new samples: {sample_path}')
            pickle.dump(dict(s_imgs=s_imgs, s_targets=s_targets), f, protocol=3)

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
        imgs = self.transform(imgs)
        prompt = random.choice(self.prompt)
        return imgs, target, prompt(self.num2str(target))

    def __str__(self):
        return f'{self.__class__.__name__} | # class: {self.n_class} | # root: {self.dataset_path} | prompt: {self.prompt}'

    @staticmethod
    def _split_warning(dataset_name, split, state):
        if split is not state:
            print(f'{colored("[DATASET_WARNING]", "red")} {dataset_name} is not supported "split" argument.')
