from collections import defaultdict

from PIL.Image import fromarray
from torch.utils.data import Dataset
from torchvision.datasets import PCAM as TorchPCAM
from torchvision.transforms import transforms

from . import VLMDataset, PCAM_CLASS_NAME


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
        target = t_dataset.h5py.File(t_dataset._base_folder / t_dataset._FILES[t_dataset._split]['targets'][0])['y'][:, 0, 0, 0]
        for i in range(len(img)):
            train_data_dict[target[i]].append(str(img[i]))
        return train_data_dict

    @staticmethod
    def loader(data):
        return fromarray(data).convert("RGB")


if __name__ == '__main__':
    ds = PCam('/data', transform=transforms.ToTensor())

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name)
