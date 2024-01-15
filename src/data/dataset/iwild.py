import os

import numpy as np
import wilds
from torch.utils.data import Dataset

from src.data.dataset import VLMDataset, IWILDCAM_CLASS_NAME

ORIGINAL_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'{name} in the wild.',
]


class IWildCam(VLMDataset, Dataset):
    dataset_path = 'iwildcam_v2.0'
    n_class = 182

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        assert split in ['train', 'val', 'test', 'id_val', 'id_test']
        dataset = wilds.get_dataset(dataset='iwildcam', root_dir=root)
        class_name_list = IWILDCAM_CLASS_NAME
        super().__init__(root, *self._imgs_targets(dataset, split, root), class_name_list, transform, target_transform,
                         n_shot)

    def _imgs_targets(self, dataset, split, root):
        split_mask = dataset.split_array == dataset.split_dict[split]
        split_idx = np.where(split_mask)[0]

        imgs = dataset._input_array[split_idx]
        for i in range(len(imgs)):
            imgs[i] = os.path.join(root, self.dataset_path, 'train', imgs[i])
        targets = dataset._y_array[split_idx].numpy()

        return imgs, targets

    @property
    def prompt(self):
        return ORIGINAL_PROMPT


if __name__ == '__main__':
    import torchvision.transforms as tf

    ds = IWildCam(root='/data', split='id_test', transform=tf.ToTensor())
    print(len(ds))
    print(np.unique(ds.targets))
    print(ds.imgs[0])
