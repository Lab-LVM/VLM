from torch.utils.data import Dataset
from torchvision.datasets import DTD as TorchDTD

from . import VLMClassificationDataset, DESCRIBABLE_TEXTURES_CLASS_NAME, DESCRIBABLE_TEXTURES_PROMPT


class DescribableTextures(VLMClassificationDataset, Dataset):
    dataset_path = 'dtd'
    n_class = 47

    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        dataset = TorchDTD(root, split)
        super().__init__(root, *self._imgs_targets(dataset), DESCRIBABLE_TEXTURES_CLASS_NAME, transform,
                         target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = dataset._image_files
        targets = dataset._labels
        return imgs, targets

    @property
    def prompt(self):
        return DESCRIBABLE_TEXTURES_PROMPT
