from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import Flowers102 as TorchFlowers102
from torchvision.transforms import transforms

from . import VLMDataset, FLOWERS102_CLASS_NAME


class Flowers102(VLMDataset, Dataset):
    dataset_path = 'flowers-102'
    n_class = 102

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchFlowers102(root, split)
        class_name_list = FLOWERS102_CLASS_NAME
        super().__init__(root, dataset._image_files, dataset._labels, class_name_list, transform, target_transform,
                         n_shot)

    @property
    def prompt(self):
        return [
            lambda c: 'a photo of a {c}, a type of flower.'
        ]

    def _data_dict(self):
        train_dataset = TorchFlowers102(self.root, 'train')
        val_dataset = TorchFlowers102(self.root, 'val')
        train_dataset._image_files.extend(val_dataset._image_files)
        train_dataset._labels.extend(val_dataset._labels)

        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset)):
            train_data_dict[train_dataset._labels[i]].append(str(train_dataset._image_files[i]))
        return train_data_dict


if __name__ == '__main__':
    ds = Flowers102('/data/vlm', transform=transforms.ToTensor(), n_shot=3)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("pink primrose")}, {ds.num2str(data[1])}')
