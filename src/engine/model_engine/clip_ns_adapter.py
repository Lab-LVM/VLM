import gc

import torch
from termcolor import colored
from torchmetrics import Accuracy
from tqdm import tqdm

from ..feature_engine import ClassificationFeatureEngine
from ..train_engine import TrainEngine
from ...data import create_dataloader
from ...utils.registry import register_task_engine, register_train_engine, register_feature_engine


@register_feature_engine
class CLIP_NSAdapterClassificationFeatureEngine(ClassificationFeatureEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_task_engine
class CLIP_NSAdapterTaskEngine:
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        self.cfg = cfg
        self.fabric = fabric
        self.device = fabric.device
        self.model = model
        self.train_dataset = train_dataset
        self.logging_interval = cfg.train.log_interval
        self.feature_engine = CLIP_NSAdapterClassificationFeatureEngine(cfg, fabric, model, tokenizer, train_dataset,
                                                                        val_dataset)

        self.loader = create_dataloader(self.cfg, val_dataset, is_train=False)

        self.metric = Accuracy('multiclass', num_classes=self.cfg.dataset.num_classes).to(self.device)

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
    def available_task(self):
        return ['classification']

    @torch.no_grad()
    def classification(self, **kwargs):
        self.metric.reset()
        total = len(self.loader) - 1

        self.model.eval()
        for i, data in tqdm(enumerate(self.loader), total=total):
            x, y = data

            x = x.to(self.device).to(memory_format=torch.channels_last)
            y = y.to(self.device)

            with self.fabric.autocast():
                prob = self.model.classifier(self.model.vision_adapter(self.model.encode_image(x)))
            self.metric.update(prob, y)

        self.metric.prefix = 'clip_classification'
        return self._output

    @property
    def _output(self):
        return {self.metric.prefix: f'{self.metric.compute().item() * 100:.3f}'}


@register_train_engine
class CLIP_NSAdapterTrainEngine(TrainEngine):
    def __init__(self, cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs):
        super().__init__(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs)

    def iterate(self, model, data, criterion):  # for additional classifier
        x, y = data

        x = x.to(self.device).to(memory_format=torch.channels_last)
        y = y.to(self.device)

        with self.fabric.autocast():
            with torch.no_grad():
                feature = model.encode_image(x)
            prob = model.classifier(model.vision_adapter(feature))
            loss = criterion(prob, y)

        return loss, prob, y

    def __call__(self, *args, **kwargs):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train_loader.sampler.set_epoch(epoch) if self.distributed else None

            train_metrics = self.train(epoch)
            self._distribute_bn()
            self.scheduler.step(epoch + 1)

            self._save(epoch, train_metrics[self.cm])
            self._log(train_metrics, {}, epoch)
            self.fabric.call('on_epoch', self.cm, self.best_metric, self.best_epoch)


@register_task_engine
class CLIP_BaseAdapterTaskEngine(CLIP_NSAdapterTaskEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_train_engine
class CLIP_BaseAdapterTrainEngine(CLIP_NSAdapterTrainEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_task_engine
class CLIP_BaseAdapter2TaskEngine(CLIP_NSAdapterTaskEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_train_engine
class CLIP_BaseAdapter2TrainEngine(CLIP_NSAdapterTrainEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_task_engine
class CLIP_NSAdapter2TaskEngine(CLIP_NSAdapterTaskEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_train_engine
class CLIP_NSAdapter2TrainEngine(CLIP_NSAdapterTrainEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
