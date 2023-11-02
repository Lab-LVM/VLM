from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet as TorchOxfordIIITPet
from torchvision.transforms import transforms

from . import VLMDataset, OXFORD_IIIT_PETS_CLASS_NAME


class OxfordIIITPet(VLMDataset, Dataset):
    dataset_path = 'oxford-iiit-pet'
    n_class = 37

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchOxfordIIITPet(root, split)
        class_name_list = OXFORD_IIIT_PETS_CLASS_NAME
        super().__init__(root, dataset._images, dataset._labels, class_name_list, transform, target_transform, n_shot)

    @property
    def prompt(self):
        return [
            lambda c: 'a photo of a {c}, a type of pet.'
        ]

    def _data_dict(self):
        train_dataset = TorchOxfordIIITPet(self.root, 'trainval')
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset._images)):
            train_data_dict[train_dataset._labels[i]].append(str(train_dataset._images[i]))
        return train_data_dict


if __name__ == '__main__':
    ds = OxfordIIITPet('/data/vlm', transform=transforms.ToTensor(), n_shot=3)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("Abyssinian")}, {ds.num2str(data[1])}')
