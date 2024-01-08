import copy

import torch
from tqdm import tqdm

from ..feature_engine import ClassificationFeatureEngine
from ..task_engine import TaskEngine
from ..train_engine import TrainEngine
from ...data import create_dataloader
from ...utils.registry import register_task_engine, register_train_engine, register_feature_engine


@register_feature_engine
class LPClassificationFeatureEngine(ClassificationFeatureEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_task_engine
class LPTaskEngine(TaskEngine):
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        feature_engine = LPClassificationFeatureEngine(cfg, fabric, model, tokenizer, train_dataset, val_dataset)
        super().__init__(feature_engine)

        self.cfg = cfg
        self.fabric = fabric
        self.device = fabric.device
        self.model = model
        self.val_dataset = val_dataset
        self.logging_interval = cfg.train.log_interval

    @property
    def available_task(self):
        return ['classification_linear_prob']

    @torch.no_grad()
    def classification_linear_prob(self, **kwargs):
        self.metric.reset()
        val_loader = create_dataloader(self.cfg, copy.deepcopy(self.val_dataset), is_train=False)
        self.model.eval()

        for data in tqdm(val_loader, total=len(val_loader), desc=f'LinearProb'):
            x, y = map(lambda x: x.to(self.device), data)
            x = x.to(memory_format=torch.channels_last)

            with self.fabric.autocast():
                image_features = self.model.encode_image(x)
                prob = self.model.classifier(image_features)

            if hasattr(self.val_dataset, 'project_logits'):
                prob = self.val_dataset.project_logits(prob)

            self.metric.update(prob, y)

        self.metric.prefix = 'clip_linear_prob'
        return self._output


@register_train_engine
class LPTrainEngine(TrainEngine):
    def __init__(self, cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs, **kwargs):
        super().__init__(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs)

    def iterate(self, model, data, criterion):
        x, y = map(lambda x: x.to(self.device), data)
        x = x.to(memory_format=torch.channels_last)

        with self.fabric.autocast():
            with torch.no_grad():
                image_features = self.model.encode_image(x)

            prob = self.model.classifier(image_features)
            loss = criterion(prob, y)

        return loss, prob, y

    def _model_train(self):
        self.model.eval()
        self.model.classifier.train()

    def _model_eval(self):
        self.model.eval()
