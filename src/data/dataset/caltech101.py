import os

from torch.utils.data import Dataset
from torchvision.datasets import Caltech101 as TorchCaltech101
from torchvision.transforms import transforms

from src.data.dataset import VLMDataset, CALTECH101_CLASS_NAME


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

    @staticmethod
    def set_prompt():
        prompt = ['a photo of a {}.',
                  'a painting of a {}.',
                  'a plastic {}.',
                  'a sculpture of a {}.',
                  'a sketch of a {}.',
                  'a tattoo of a {}.',
                  'a toy {}.',
                  'a rendition of a {}.',
                  'a embroidered {}.',
                  'a cartoon {}.',
                  'a {} in a video game.',
                  'a plushie {}.',
                  'a origami {}.',
                  'art of a {}.',
                  'graffiti of a {}.',
                  'a drawing of a {}.',
                  'a doodle of a {}.',
                  'a photo of the {}.',
                  'a painting of the {}.',
                  'the plastic {}.',
                  'a sculpture of the {}.',
                  'a sketch of the {}.',
                  'a tattoo of the {}.',
                  'the toy {}.',
                  'a rendition of the {}.',
                  'the embroidered {}.',
                  'the cartoon {}.',
                  'the {} in a video game.',
                  'the plushie {}.',
                  'the origami {}.',
                  'art of the {}.',
                  'graffiti of the {}.',
                  'a drawing of the {}.',
                  'a doodle of the {}.',
                  ]
        return prompt


if __name__ == '__main__':
    ds = Caltech101('/data/vlm', transform=transforms.ToTensor(), n_shot=4)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("background")}, {ds.num2str(data[1])}')
