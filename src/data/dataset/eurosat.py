from torch.utils.data import Dataset
from torchvision.datasets import EuroSAT as TorchEuroSAT
from torchvision.transforms import transforms

from . import VLMDataset, EUROSAT_CLASS_NAME


class EuroSAT(VLMDataset, Dataset):
    dataset_path = 'eurosat'
    n_class = 10

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = TorchEuroSAT(root)
        class_name_list = EUROSAT_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return [
            lambda c: f'a centered satellite photo of {c}.',
            lambda c: f'a centered satellite photo of a {c}.',
            lambda c: f'a centered satellite photo of the {c}.',
        ]


if __name__ == '__main__':
    ds = EuroSAT('/data/vlm', transform=transforms.ToTensor(), n_shot=0)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("forest")}, {ds.num2str(data[1])}')
