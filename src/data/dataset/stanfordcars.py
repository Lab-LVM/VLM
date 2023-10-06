from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import StanfordCars as TorchStanfordCars
from torchvision.transforms import transforms

from src.data.dataset import VLMDataset, STANFORDCARS_CLASS_NAME


class StanfordCars(VLMDataset, Dataset):
    dataset_path = 'stanford_cars'
    n_class = 196

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchStanfordCars(root, split)
        class_name_list = STANFORDCARS_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs, targets = list(), list()
        for i in range(len(dataset)):
            _data = dataset._samples[i]
            imgs.append(_data[0])
            targets.append(_data[1])
        return imgs, targets

    @staticmethod
    def set_prompt():
        prompt = [
            'a photo of a {}.',
            'a photo of the {}.',
            'a photo of my {}.',
            'i love my {}!',
            'a photo of my dirty {}.',
            'a photo of my clean {}.',
            'a photo of my new {}.',
            'a photo of my old {}.',
        ]
        return prompt

    def _data_dict(self):
        train_dataset = TorchStanfordCars(self.root, 'train')
        train_data_dict = defaultdict(list)
        _imgs, _targets = self._imgs_targets(train_dataset)

        for i in range(len(_imgs)):
            train_data_dict[_targets[i]].append(str(_imgs[i]))
        return train_data_dict


if __name__ == '__main__':
    ds = StanfordCars('/data/vlm', transform=transforms.ToTensor(), n_shot=3)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("AM General Hummer SUV 2000")}, {ds.num2str(data[1])}')
