import os
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import ImageNet as TorchImagenet
from torchvision.transforms import transforms

from src.data.dataset import VLMDataset, IMAGENET_CLASS_NAME


class ImageNet(VLMDataset, Dataset):
    dataset_path = 'imageNet'
    n_class = 1000

    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        dataset = TorchImagenet(os.path.join(root, self.dataset_path), split)
        class_name_list = IMAGENET_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset), class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @staticmethod
    def set_prompt():
        prompt = ["itap of a {}.",
                  "a bad photo of the {}.",
                  "a origami {}.",
                  "a photo of the large {}.",
                  "a {} in a video game.",
                  "art of the {}.",
                  "a photo of the small {}.",
                  ]
        return prompt

    def _data_dict(self):
        train_dataset = TorchImagenet(os.path.join(self.root, self.dataset_path), 'train')
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset.imgs)):
            train_data_dict[train_dataset.targets[i]].append(train_dataset.imgs[i][0])

        return train_data_dict


if __name__ == '__main__':
    ds = ImageNet('/data/vlm', transform=transforms.ToTensor(), n_shot=0)
    ds.sampling(1)
    print(len(ds))
    data = next(iter(ds))
    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("tench")}, {ds.num2str(data[1])}')
