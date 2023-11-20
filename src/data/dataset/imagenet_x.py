import json
import os
from abc import ABC
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from . import VLMDataset, IMAGENET_CLASS_NAME

imagenet_a_class_number = [
    6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108,
    110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307,
    308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386,
    397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483,
    486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606,
    607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758,
    763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845,
    847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954,
    956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988
]

imagenet_r_class_number = [
    1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113,
    122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207,
    208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289,
    291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344,
    347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437,
    441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594,
    596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820,
    824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951,
    953, 954, 957, 963, 965, 967, 980, 981, 983, 988
]


class ImageNetX(VLMDataset, Dataset, ABC):
    dataset_path = 'imageNet-X'
    n_class = 1000
    class_number = range(0, 1000)

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = ImageFolder(os.path.join(root, self.dataset_path))
        class_name_list = [IMAGENET_CLASS_NAME[i] for i in self.class_number]
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


class ImageNetR(ImageNetX):
    dataset_path = 'imageNet-R'
    n_class = 200
    class_number = imagenet_r_class_number


class ImageNetA(ImageNetX):
    dataset_path = 'imageNet-A'
    n_class = 200
    class_number = imagenet_a_class_number


class ImageNetSketch(ImageNetX):
    dataset_path = 'imageNet-Sketch'


class ImageNetV2(VLMDataset, Dataset, ABC):
    dataset_path = 'imageNet-V2'
    n_class = 1000

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        imgs, targets = list(), list()
        for sample in glob(os.path.join(root, self.dataset_path, '*/*')):
            imgs.append(sample)
            targets.append(int(os.path.basename(os.path.dirname(sample))))
        class_name_list = IMAGENET_CLASS_NAME
        super().__init__(root, imgs, targets, class_name_list, transform, target_transform, n_shot)

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


class ObjectNet(VLMDataset, Dataset):
    dataset_path = 'objectnet-1.0'
    n_class = 113

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        self.folder_to_ids = self.get_metadata(root)

        imgs, targets, class_name_list = list(), list(), list()
        target_number = 0
        for class_name, idxs in self.folder_to_ids.items():
            class_img = glob(os.path.join(root, self.dataset_path, 'images', class_name, '*'))
            imgs.extend(class_img)
            targets.extend([target_number for _ in range(len(class_img))])
            class_name = ' or '.join([IMAGENET_CLASS_NAME[idx] for idx in idxs])
            class_name_list.append(class_name)
            target_number += 1

        super().__init__(root, imgs, targets, class_name_list, transform, target_transform, n_shot)

    def get_metadata(self, root):
        metadata = Path(os.path.join(root, self.dataset_path, 'mappings'))
        with open(metadata / 'folder_to_objectnet_label.json', 'r') as f:
            folder_map = json.load(f)
            folder_map = {v: k for k, v in folder_map.items()}
        with open(metadata / 'objectnet_to_imagenet_1k.json', 'r') as f:
            objectnet_map = json.load(f)

        with open(metadata / 'pytorch_to_imagenet_2012_id.json', 'r') as f:
            pytorch_map = json.load(f)
            pytorch_map = {v: k for k, v in pytorch_map.items()}

        with open(metadata / 'imagenet_to_label_2012_v2', 'r') as f:
            imagenet_map = {v.strip(): str(pytorch_map[i]) for i, v in enumerate(f)}

        folder_to_ids = dict()
        for objectnet_name, imagenet_names in objectnet_map.items():
            imagenet_names = imagenet_names.split('; ')
            imagenet_ids = [int(imagenet_map[imagenet_name]) for imagenet_name in imagenet_names]
            folder_to_ids[folder_map[objectnet_name]] = imagenet_ids

        return folder_to_ids

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

    def to_imageNet_logits(self, logits):
        new_logits = torch.zeros((logits.size(0), self.n_class), device=logits.device)

        for class_name, idxs in self.folder_to_ids.items():
            new_logits[:, self.str2num(class_name)] = logits[:, idxs].mean(-1)

        return new_logits


if __name__ == '__main__':
    for d_class in [ImageNetR, ImageNetA, ImageNetV2, ImageNetSketch, ObjectNet]:
        print(d_class.__name__)
        ds = d_class('/data', transform=transforms.ToTensor())
        data = next(iter(ds))
        print(data[0].shape, data[1])
        print(ds.class_name[:5])
        print(f'{ds.str2num("tench")}, {ds.num2str(data[1])}')

        logits = torch.rand(100, 1000)
        new_logits = ds.to_imageNet_logits(logits)
        print(new_logits.shape)
