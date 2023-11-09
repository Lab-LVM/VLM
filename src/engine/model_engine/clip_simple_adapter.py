import torch

from .clip import CLIPTaskEngine
from ..feature_engine import ClassificationFeatureEngine
from ..train_engine import TrainEngine
from ...utils.registry import register_task_engine, register_train_engine, register_feature_engine


@register_feature_engine
class CLIP_SimpleAdapterClassificationFeatureEngine(ClassificationFeatureEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_task_engine
class CLIP_SimpleAdapterTaskEngine(CLIPTaskEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_train_engine
class CLIP_SimpleAdapterTrainEngine(TrainEngine):
    def __init__(self, cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs):
        super().__init__(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs)
        self.train_loader.dataset.setup_prompt_transform()

    def iterate(self, model, data, criterion):
        x, y, ra_prompt = data

        x = x.to(self.device).to(memory_format=torch.channels_last)
        onehot_y = torch.arange(x.shape[0]).long().to(self.device)
        ra_prompt = self._tokenize(ra_prompt).to(self.device)

        with self.fabric.autocast():
            logits_per_image, logits_per_text = model(x, ra_prompt)
            loss = criterion(logits_per_image, logits_per_text)

        return loss, logits_per_image, onehot_y

    def iterate_ra2(self, model, data, criterion):
        x, ra_x, y, prompt, ra_prompt = data

        x = torch.concat([x, ra_x])
        prompt.extend(ra_prompt)

        x = x.to(self.device).to(memory_format=torch.channels_last)
        onehot_y = torch.arange(x.shape[0]).long().to(self.device)
        prompt = self._tokenize(prompt).to(self.device)

        with self.fabric.autocast():
            logits_per_image, logits_per_text = model(x, prompt)
            loss = criterion(logits_per_image, logits_per_text)

        return loss, logits_per_image, onehot_y

    def __call__(self, *args, **kwargs):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train_loader.sampler.set_epoch(epoch) if self.distributed else None

            train_metrics = self.train(epoch)
            self._distribute_bn()
            self.scheduler.step(epoch + 1)

            self._save(epoch, train_metrics[self.cm])
            self._log(train_metrics, {}, epoch)
            self.fabric.call('on_epoch', self.cm, self.best_metric, self.best_epoch)
