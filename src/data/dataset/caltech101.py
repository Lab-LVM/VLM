import os

from torch.utils.data import Dataset
from torchvision.datasets import Caltech101 as TorchCaltech101
from torchvision.transforms import transforms

from . import VLMDataset, CALTECH101_CLASS_NAME


class Caltech101(VLMDataset, Dataset):
    dataset_path = 'caltech101'
    n_class = 101

    def __init__(self, root, split='category', transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, 'category')
        dataset = TorchCaltech101(root, split)
        class_name_list = CALTECH101_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = []
        targets = []
        for i in range(len(dataset)):
            imgs.append(os.path.join(
                dataset.root,
                '101_ObjectCategories',
                dataset.categories[dataset.y[i]],
                f'image_{dataset.index[i]:04d}.jpg',
            ))
            targets.append(dataset.y[i])
        return imgs, targets

    @property
    def prompt(self):
        return [
            lambda c: f'a photo of a {c}.',
            lambda c: f'a painting of a {c}.',
            lambda c: f'a plastic {c}.',
            lambda c: f'a sculpture of a {c}.',
            lambda c: f'a sketch of a {c}.',
            lambda c: f'a tattoo of a {c}.',
            lambda c: f'a toy {c}.',
            lambda c: f'a rendition of a {c}.',
            lambda c: f'a embroidered {c}.',
            lambda c: f'a cartoon {c}.',
            lambda c: f'a {c} in a video game.',
            lambda c: f'a plushie {c}.',
            lambda c: f'a origami {c}.',
            lambda c: f'art of a {c}.',
            lambda c: f'graffiti of a {c}.',
            lambda c: f'a drawing of a {c}.',
            lambda c: f'a doodle of a {c}.',
            lambda c: f'a photo of the {c}.',
            lambda c: f'a painting of the {c}.',
            lambda c: f'the plastic {c}.',
            lambda c: f'a sculpture of the {c}.',
            lambda c: f'a sketch of the {c}.',
            lambda c: f'a tattoo of the {c}.',
            lambda c: f'the toy {c}.',
            lambda c: f'a rendition of the {c}.',
            lambda c: f'the embroidered {c}.',
            lambda c: f'the cartoon {c}.',
            lambda c: f'the {c} in a video game.',
            lambda c: f'the plushie {c}.',
            lambda c: f'the origami {c}.',
            lambda c: f'art of the {c}.',
            lambda c: f'graffiti of the {c}.',
            lambda c: f'a drawing of the {c}.',
            lambda c: f'a doodle of the {c}.',
        ]


if __name__ == '__main__':
    ds = Caltech101('/data/vlm', transform=transforms.ToTensor(), n_shot=4)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("background")}, {ds.num2str(data[1])}')
