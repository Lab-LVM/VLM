from time import time

import torch

from ..feature_engine import ClassificationFeatureEngine
from ..task_engine import TaskEngine
from ..train_engine import TrainEngine
from ...utils.registry import register_task_engine, register_train_engine, register_feature_engine


@register_feature_engine
class CLIPClassificationFeatureEngine(ClassificationFeatureEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_task_engine
class CLIPTaskEngine(TaskEngine):
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        feature_engine = CLIPClassificationFeatureEngine(cfg, fabric, model, tokenizer, train_dataset, val_dataset)
        super().__init__(feature_engine)

    @property
    def available_task(self):
        return ['classification_zeroshot']

    def classification_zeroshot(self, **kwargs):
        self.feature_engine.sampling(0)
        self.metric.reset()

        text_classifier = self.feature_engine.build_text_classifier()
        qry_features, qry_labels = self.feature_engine.build_query_set()

        logits = 100. * qry_features @ text_classifier.mT

        self.metric.update(logits, qry_labels)
        self.metric.prefix = 'clip_zeroshot'
        return self._output


@register_train_engine
class CLIPTrainEngine(TrainEngine):
    def __init__(self, cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs, **kwargs):
        super().__init__(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs)
        assert isinstance(optimizer, torch.optim.LBFGS), 'Only LBFGS optimizer is supported.'

    def train(self, epoch):
        self._reset_metric()
        self._model_train()

        total = len(self.train_loader) - 1

        start = time()
        for i, data in enumerate(self.train_loader):
            x, y = map(lambda x: x.to(self.device), data)
            x = x.to(memory_format=torch.channels_last)
            with torch.no_grad():
                image_features = self.model.encode_image(x)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            def closure():
                self.optimizer.zero_grad()

                prob = self.model.adapter(image_features)
                loss = self.train_criterion(prob, y)
                self.fabric.backward(loss)
                return loss

            self.optimizer.step(closure)

            loss = closure()
            self.losses.update(loss)

            tnow = time()
            duration, start = tnow - start, tnow
            if i % self.logging_interval == 0 or i == total:
                lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                self.fabric.call('on_train', epoch, i, total, loss, lr, duration, self.sample_size)

        return {'loss': self.losses.compute().item()}

    @torch.no_grad()
    def eval(self, epoch):
        self._reset_metric()
        self._model_eval()

        total = len(self.val_loader) - 1

        for i, data in enumerate(self.val_loader):
            x, y = map(lambda x: x.to(self.device), data)
            x = x.to(memory_format=torch.channels_last)

            image_features = self.model.encode_image(x)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prob = self.model.adapter(image_features)
            loss = self.val_criterion(prob, y)
            self._update_metric(loss, prob, y)

            if i % self.logging_interval == 0 or i == total - 1:
                self.fabric.call('on_eval', self._metrics(), epoch, i, total)

        return self._metrics()

    def _model_train(self):
        self.model.eval()

    def _model_eval(self):
        self.model.eval()
