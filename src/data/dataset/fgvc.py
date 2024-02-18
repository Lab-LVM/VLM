from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import FGVCAircraft as TorchFGVCAircraft

from . import VLMClassificationDataset, FGVC_CLASS_NAME, FGVC_PROMPT


class FGVCAircraft(VLMClassificationDataset, Dataset):
    dataset_path = 'fgvc-aircraft-2013b'
    n_class = 100

    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        dataset = TorchFGVCAircraft(root, split)
        super().__init__(root, dataset._image_files, dataset._labels, FGVC_CLASS_NAME, transform, target_transform,
                         n_shot)

    @property
    def prompt(self):
        return FGVC_PROMPT
