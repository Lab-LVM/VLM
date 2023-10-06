from collections import defaultdict

from torch.utils.data import Dataset
from torchvision.datasets import FGVCAircraft as TorchFGVCAircraft
from torchvision.transforms import transforms

from src.data.dataset import VLMDataset, FGVC_CLASS_NAME


class FGVCAircraft(VLMDataset, Dataset):
    dataset_path = 'fgvc-aircraft-2013b'
    n_class = 100

    def __init__(self, root, split='val', transform=None, target_transform=None, n_shot=0):
        dataset = TorchFGVCAircraft(root, split)
        class_name_list = FGVC_CLASS_NAME
        super().__init__(root, dataset._image_files, dataset._labels, class_name_list, transform, target_transform, n_shot)

    @staticmethod
    def set_prompt():
        prompt = [
            'a photo of a {}, a type of aircraft.',
            'a photo of the {}, a type of aircraft.',
        ]
        return prompt

    def _data_dict(self):
        train_dataset = TorchFGVCAircraft(self.root, 'train')
        train_data_dict = defaultdict(list)
        for i in range(len(train_dataset._image_files)):
            train_data_dict[train_dataset._labels[i]].append(train_dataset._image_files[i])
        return train_data_dict


if __name__ == '__main__':
    ds = FGVCAircraft('/data/vlm', transform=transforms.ToTensor(), n_shot=3)

    data = next(iter(ds))

    print(data[0].shape, data[1])
    print(ds.class_name[:5])
    print(f'{ds.str2num("707-320")}, {ds.num2str(data[1])}')
