import gc
from abc import ABC, abstractmethod

import torch
from termcolor import colored
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

from . import ClassificationFeatureEngine
from ..utils.registry import register_task_engine


class TaskEngine(ABC):
    name = 'ModelName'

    def __call__(self, **kwargs):
        output = dict()
        for task_name in self.available_task:
            metric = self.task(task_name, **kwargs)
            output.update(metric)

            torch.cuda.empty_cache()
            gc.collect()

        return output

    def task(self, task_name, **kwargs):
        try:
            return self.__getattribute__(task_name)(**kwargs)
        except NotImplementedError:
            self.warning(f'Available tasks are {",".join(self.available_task)}')

    def warning(self, text):
        print(f'{colored(f"Warning:[{self.__class__.__name__}]", "red")} {text}')

    @property
    @abstractmethod
    def available_task(self):
        return ['downstream_task']

    @property
    def _output(self):
        return {self.metric.prefix: f'{self.metric.compute().item() * 100:.3f}'}


class ClassificationTaskEngine(TaskEngine):
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        super().__init__()
        self.cfg = cfg
        self.fabric = fabric
        self.device = fabric.device
        self.model = model
        self.val_dataset = val_dataset
        self.logging_interval = cfg.train.log_interval

        self.feature_engine = ClassificationFeatureEngine(cfg, fabric, model, tokenizer, train_dataset, val_dataset)
        self.metric = Accuracy('multiclass', num_classes=val_dataset.n_class).to(self.device)

    @property
    def available_task(self):
        return ['classification_n_shot', 'classification_linear_prob']

    def classification_n_shot(self, n_shot=0):
        self.feature_engine.sampling(n_shot)
        self.metric.reset()

        text_classifier = self.feature_engine.build_text_classifier()
        qry_features, qry_labels = self.feature_engine.build_query_set()

        logits = self.model.logit_scale.exp() * qry_features @ text_classifier.mT

        if hasattr(self.val_dataset, 'project_logits'):
            logits = self.val_dataset.project_logits(logits)

        self.metric.update(logits, qry_labels)
        self.metric.prefix = f'{n_shot}shot'
        return self._output

    @torch.no_grad()
    def classification_linear_prob(self, n_shot=None):
        self.metric.reset()
        self.model.eval()

        val_loader = DataLoader(self.val_dataset, self.cfg.train.batch_size, num_workers=4, pin_memory=True)

        for data in tqdm(val_loader, total=len(val_loader), desc=f'LinearProb'):
            x, y, _ = map(lambda x: x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x, data)
            x = x.to(memory_format=torch.channels_last)

            with self.fabric.autocast():
                prob = self.model(x)

            self.metric.update(prob, y)

        self.metric.prefix = 'linear_prob'
        return self._output


@register_task_engine
class FewshotTaskEngine(ClassificationTaskEngine):
    @property
    def available_task(self):
        return ['classification_n_shot']

    def set_n_shot(self, n_shot):
        self.n_shot = n_shot


@register_task_engine
class LPTaskEngine(ClassificationTaskEngine):
    @property
    def available_task(self):
        return ['classification_linear_prob']

