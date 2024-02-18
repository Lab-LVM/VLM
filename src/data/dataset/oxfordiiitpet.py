from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet as TorchOxfordIIITPet

from . import VLMClassificationDataset, OXFORD_IIIT_PETS_CLASS_NAME, OXFORD_IIIT_PETS_PROMPT


class OxfordIIITPet(VLMClassificationDataset, Dataset):
    dataset_path = 'oxford-iiit-pet'
    n_class = 37

    def __init__(self, root, split='test', transform=None, target_transform=None, n_shot=0):
        dataset = TorchOxfordIIITPet(root, split)
        super().__init__(root, dataset._images, dataset._labels, OXFORD_IIIT_PETS_CLASS_NAME, transform,
                         target_transform, n_shot)

    @property
    def prompt(self):
        return OXFORD_IIIT_PETS_PROMPT
