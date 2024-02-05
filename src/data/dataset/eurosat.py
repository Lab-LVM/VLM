from torch.utils.data import Dataset
from torchvision.datasets import EuroSAT as TorchEuroSAT

from . import VLMClassificationDataset, EUROSAT_CLASS_NAME, EUROSAT_PROMPT


class EuroSAT(VLMClassificationDataset, Dataset):
    dataset_path = 'eurosat'
    n_class = 10

    def __init__(self, root, split=None, transform=None, target_transform=None, n_shot=0):
        self._split_warning(self.__class__.__name__, split, None)
        dataset = TorchEuroSAT(root)
        super().__init__(root, *self._imgs_targets(dataset), EUROSAT_CLASS_NAME, transform, target_transform, n_shot)

    @staticmethod
    def _imgs_targets(dataset):
        imgs = [x[0] for x in dataset.imgs]
        targets = dataset.targets
        return imgs, targets

    @property
    def prompt(self):
        return EUROSAT_PROMPT
