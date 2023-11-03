import os
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import Country211 as TorchCountry211
from torchvision.transforms import transforms

from src.data.dataset import VLMDataset, COUNTRY211_CLASS_NAME


class Country211(VLMDataset, Dataset):
    dataset_path = 'country211'
    n_class = 211

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchCountry211(root, split)
        class_name_list = COUNTRY211_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return [
            lambda c: f'a photo i took in {c}.',
            lambda c: f'a photo i took while visiting {c}.',
            lambda c: f'a photo from my home country of {c}.',
            lambda c: f'a photo from my visit to {c}.',
            lambda c: f'a photo showing the country of {c}.',
        ]

    def _data_dict(self):
        train_dataset = TorchCountry211(os.path.join(self.root), 'train')
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset.imgs)):
            train_data_dict[train_dataset.targets[i]].append(train_dataset.imgs[i][0])

        return train_data_dict


if __name__ == '__main__':
    ds = Country211('/data', transform=transforms.ToTensor(), n_shot=0)
    print(len(ds))
    data = next(iter(ds))
    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("tench")}, {ds.num2str(data[1])}')
