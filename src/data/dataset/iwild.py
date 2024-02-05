import os

import numpy as np
import wilds
from torch.utils.data import Dataset

from . import VLMClassificationDataset, IWILDCAM_CLASS_NAME, IWILDCAM_PROMPT


class IWildCam(VLMClassificationDataset, Dataset):
    dataset_path = 'iwildcam_v2.0'
    n_class = 182

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        assert split in ['train', 'val', 'test', 'id_val', 'id_test']
        self.dataset = wilds.get_dataset(dataset='iwildcam', root_dir=root)

        super().__init__(root, *self._imgs_targets(self.dataset, split, root), IWILDCAM_CLASS_NAME, transform,
                         target_transform, n_shot)

    def _imgs_targets(self, dataset, split, root):
        split_mask = dataset.split_array == dataset.split_dict[split]
        split_idx = np.where(split_mask)[0]

        imgs = dataset._input_array[split_idx]
        for i in range(len(imgs)):
            imgs[i] = os.path.join(root, self.dataset_path, 'train', imgs[i])
        targets = dataset.y_array[split_idx].numpy()
        self.meta_data = dataset.metadata_array

        return imgs, targets

    def scoring(self, y_pred, y_true):
        return self.dataset.eval(y_pred, y_true, self.meta_data)

    @property
    def prompt(self):
        return IWILDCAM_PROMPT
