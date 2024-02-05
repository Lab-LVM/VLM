import os

import numpy as np
import wilds
from torch.utils.data import Dataset

from src.data.dataset import VLMClassificationDataset, FMOW_CLASS_NAME, FMOW_PROMPT


class FMow(VLMClassificationDataset, Dataset):
    dataset_path = 'fmow_v1.1'
    n_class = 62

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        assert split in ['train', 'val', 'test', 'id_val', 'id_test']

        self.dataset = wilds.get_dataset(dataset='fmow', root_dir=root)
        super().__init__(root, *self._imgs_targets(self.dataset, split, root), FMOW_CLASS_NAME, transform,
                         target_transform, n_shot)

    def _imgs_targets(self, dataset, split, root):
        split_mask = dataset.split_array == dataset.split_dict[split]
        split_idx = np.where(split_mask)[0]

        imgs = list()
        for img_idx in dataset.split_array[split_idx]:
            imgs.append(os.path.join(root, self.dataset_path, 'images', f'rgb_img_{int(img_idx)}.png'))

        imgs = np.array(imgs)
        targets = dataset.y_array[split_idx].numpy()
        self.meta_datas = dataset.metadata_array

        return imgs, targets

    def scoring(self, y_pred, y_true):
        return self.dataset.eval(y_pred, y_true, self.meta_datas)

    @property
    def prompt(self):
        return FMOW_PROMPT
