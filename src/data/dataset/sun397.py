from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import SUN397 as TorchSUN397
from torchvision.transforms import transforms

from . import VLMDataset, SUN397_CLASS_NAME


class SUN397(VLMDataset, Dataset):
    dataset_path = 'SUN397'
    n_class = 397

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = TorchSUN397(root)
        self.dataset = dataset
        class_name_list = SUN397_CLASS_NAME
        super().__init__(root, dataset._image_files, dataset._labels, class_name_list, transform, target_transform,
                         n_shot)

    @property
    def prompt(self):
        return [
            lambda c: f'a photo of a {c}.',
            lambda c: f'a photo of the {c}.',
        ]

    def _data_dict(self):
        data_dict = defaultdict(list)
        for i in range(len(self.dataset._image_files)):
            data_dict[self.dataset._labels[i]].append(str(self.dataset._image_files[i]))
        return data_dict


if __name__ == '__main__':
    ds = SUN397('/data/vlm', transform=transforms.ToTensor(), n_shot=0)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("abbey")}, {ds.num2str(data[1])}')
