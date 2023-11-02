from torch.utils.data import Dataset
from torchvision.datasets import DTD as TorchDTD
from torchvision.transforms import transforms

from . import VLMDataset, DESCRIBABLETEXTURES_CLASS_NAME


class DescribableTextures(VLMDataset, Dataset):
    dataset_path = 'dtd'
    n_class = 47

    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        dataset = TorchDTD(root, split)
        class_name_list = DESCRIBABLETEXTURES_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x for x in dataset._image_files]
        targets = dataset._labels
        return imgs, targets

    @property
    def prompt(self):
        return [
            lambda c: f'a photo of a {c} texture.',
            lambda c: f'a photo of a {c} pattern.',
            lambda c: f'a photo of a {c} thing.',
            lambda c: f'a photo of a {c} object.',
            lambda c: f'a photo of the {c} texture.',
            lambda c: f'a photo of the {c} pattern.',
            lambda c: f'a photo of the {c} thing.',
            lambda c: f'a photo of the {c} object.',
        ]


if __name__ == '__main__':
    ds = DescribableTextures('/data', transform=transforms.ToTensor(), n_shot=0)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("forest")}, {ds.num2str(data[1])}')
