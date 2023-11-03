from collections import defaultdict

from PIL.Image import fromarray
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100 as TorchCIFAR100
from torchvision.transforms import transforms

from . import VLMDataset, CIFAR100_CLASS_NAME


class CIFAR100(VLMDataset, Dataset):
    dataset_path = 'cifar-100-python'
    n_class = 100

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchCIFAR100(root, train=split == 'train')
        class_name_list = CIFAR100_CLASS_NAME
        super().__init__(root, dataset.data, dataset.targets, class_name_list, transform, target_transform, n_shot)

    @property
    def prompt(self):
        return [
            lambda c: f'a photo of a {c}.',
            lambda c: f'a blurry photo of a {c}.',
            lambda c: f'a black and white photo of a {c}.',
            lambda c: f'a low contrast photo of a {c}.',
            lambda c: f'a high contrast photo of a {c}.',
            lambda c: f'a bad photo of a {c}.',
            lambda c: f'a good photo of a {c}.',
            lambda c: f'a photo of a small {c}.',
            lambda c: f'a photo of a big {c}.',
            lambda c: f'a photo of the {c}.',
            lambda c: f'a blurry photo of the {c}.',
            lambda c: f'a black and white photo of the {c}.',
            lambda c: f'a low contrast photo of the {c}.',
            lambda c: f'a high contrast photo of the {c}.',
            lambda c: f'a bad photo of the {c}.',
            lambda c: f'a good photo of the {c}.',
            lambda c: f'a photo of the small {c}.',
            lambda c: f'a photo of the big {c}.',
        ]

    def _data_dict(self):
        train_dataset = TorchCIFAR100(self.root, train=True)
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset.data)):
            train_data_dict[train_dataset.targets[i]].append(str(train_dataset.data[i]))
        return train_data_dict

    @staticmethod
    def loader(data):
        return fromarray(data)


if __name__ == '__main__':
    ds = CIFAR100('/data', transform=transforms.ToTensor())

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("Abyssinian")}, {ds.num2str(data[1])}')
