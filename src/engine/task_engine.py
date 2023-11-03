import gc
from abc import ABC, abstractmethod

import torch
from termcolor import colored
from torchmetrics import Accuracy

from .feature_engine import FeatureEngine


class TaskEngine(ABC):
    def __init__(self, feature_engine: FeatureEngine):
        self.feature_engine = feature_engine
        self.metric = Accuracy('multiclass', num_classes=feature_engine.num_class).to(feature_engine.device)

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
