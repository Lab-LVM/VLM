import abc
import copy
from pathlib import Path

import hydra
import torch
from torch.nn.functional import normalize
from tqdm import tqdm

from ..data import create_dataloader


class FeatureEngine(abc.ABC):
    pass


class ClassificationFeatureEngine(FeatureEngine):
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        self.cfg = cfg
        self.fabric = fabric
        self.cache = cfg.cache
        self.model_name = cfg.model.model_name
        self.dataset_name = cfg.dataset.name
        self.num_class = train_dataset.n_class
        self.cache_path = self._make_cache_path()

        self.model = model
        self.tokenizer = tokenizer
        self.device = fabric.device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.text_classifier = None
        self.sup_features = None
        self.sup_labels = None
        self.qry_features = None
        self.qry_labels = None

    @torch.no_grad()
    def build_text_classifier(self):
        file_name = f'{self.dataset_name}_text_classifier'
        text_classifier = self._open_file(file_name)
        if not isinstance(text_classifier, str):
            self.text_classifier = text_classifier
            return self.text_classifier

        text_classifier = list()
        self.model.eval()
        for class_name in tqdm(self.train_dataset.class_name, desc='Build Text Classifier'):
            text = [p(class_name) for p in self.train_dataset.prompt]
            text_input = self.tokenizer(text, padding='max_length', return_attention_mask=False, return_tensors='pt')['input_ids'].to(self.device)

            with self.fabric.autocast():
                text_feature = self.model.module.encode_text(text_input)

            text_feature = normalize(text_feature, dim=-1).mean(0)
            text_feature /= text_feature.norm()
            text_classifier.append(text_feature)

        self.text_classifier = torch.stack(text_classifier, dim=0).to(self.device)

        self._save_file(self.text_classifier, file_name)
        return self.text_classifier

    @torch.no_grad()
    def build_support_set(self):
        if self.train_dataset.n_shot == 0:
            return None, None

        file_name = f'{self.dataset_name}_support_{self.train_dataset.n_shot}s_feature'
        sets = self._open_file(file_name)
        if not isinstance(sets, str):
            self.sup_features, self.sup_labels = sets
            return self.sup_features, self.sup_labels

        features, labels = list(), list()
        loader = create_dataloader(self.cfg, copy.deepcopy(self.train_dataset), is_train=False)

        self.model.eval()
        for data in tqdm(loader, total=len(loader), desc=f'Build {self.train_dataset.n_shot}-Shot Support Set'):
            x, y, _ = map(lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, data)
            x = x.to(memory_format=torch.channels_last)

            with self.fabric.autocast():
                image_features = self.model.module.encode_image(x)

            features.append(image_features.detach().cpu())
            labels.append(y.detach().cpu())

        features = torch.cat(features, dim=0).to(self.device)
        self.sup_features = normalize(features, dim=-1)
        self.sup_labels = torch.cat(labels).to(self.device)

        self._save_file((self.sup_features, self.sup_labels), file_name)
        return self.sup_features, self.sup_labels

    @torch.no_grad()
    def build_query_set(self):
        file_name = f'{self.dataset_name}_query_feature'
        sets = self._open_file(file_name)
        if not isinstance(sets, str):
            self.qry_features, self.qry_labels = sets
            return self.qry_features, self.qry_labels

        features, labels = list(), list()
        loader = create_dataloader(self.cfg, copy.deepcopy(self.val_dataset), is_train=False)

        self.model.eval()
        for data in tqdm(loader, total=len(loader), desc=f'Build Query Set'):
            x, y, _ = map(lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, data)
            x = x.to(memory_format=torch.channels_last)

            with self.fabric.autocast():
                image_features = self.model.module.encode_image(x)

            features.append(image_features.detach().cpu())
            labels.append(y.detach().cpu())

        features = torch.cat(features, dim=0).to(self.device)
        self.qry_features = normalize(features, dim=-1)
        self.qry_labels = torch.cat(labels).to(self.device)

        self._save_file((self.qry_features, self.qry_labels), file_name)
        return self.qry_features, self.qry_labels

    def get_all_features(self):
        self.build_text_classifier() if self.text_classifier is None else None
        self.build_support_set() if self.sup_features is None else None
        self.build_query_set() if self.qry_features is None else None

        return dict(
            text_classifier=self.text_classifier,
            sup_features=self.sup_features,
            sup_labels=self.sup_labels,
            qry_features=self.qry_features,
            qry_labels=self.qry_labels,
        )

    def sampling(self, n_shot):
        if self.train_dataset.n_shot == n_shot:
            return
        self.train_dataset.sampling(n_shot)
        self.build_support_set()

    def _make_cache_path(self):
        path = Path(hydra.utils.get_original_cwd()) / 'cache' / self.model_name
        if not path.exists():
            path.mkdir(parents=True)
        return path

    def _open_file(self, file_name):
        if not self.cache:
            return ''

        path = self.cache_path / file_name
        if path.exists():
            return torch.load(path, map_location=self.device)
        return str(path)

    def _save_file(self, obj, file_name):
        if self.cache:
            torch.save(obj, self.cache_path / file_name)
