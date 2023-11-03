import os

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from . import VLMDataset, UCF101_CLASS_NAME


class UCF101(VLMDataset, Dataset):
    dataset_path = 'UCF-101-midframes'
    n_class = 101

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = ImageFolder(os.path.join(root, self.dataset_path))
        class_name_list = UCF101_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return [
            lambda c: f'a photo of a person {c}.',
            lambda c: f'a photo of a person using {c}.',
            lambda c: f'a photo of a person doing {c}.',
            lambda c: f'a photo of a person during {c}.',
            lambda c: f'a photo of a person performing {c}.',
            lambda c: f'a photo of a person practicing {c}.',
        ]


if __name__ == '__main__':
    ds = UCF101('/data', transform=transforms.ToTensor(), n_shot=0)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("Archery")}, {ds.num2str(data[1])}')
