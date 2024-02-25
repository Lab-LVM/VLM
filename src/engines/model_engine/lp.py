from ..train_engine import TrainEngine
from ...utils.registry import register_train_engine


@register_train_engine
class LPTrainEngine(TrainEngine):
    def __init__(self, cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs, **kwargs):
        super().__init__(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs)

    def __call__(self, *args, **kwargs):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train_loader.dataset.set_feature(epoch) if hasattr(self.train_loader.dataset, 'set_feature') else None
            self.train_loader.sampler.set_epoch(epoch) if self.distributed else None

            train_metrics = self.train(epoch)
            self._distribute_bn()
            self.scheduler.step(epoch + 1)

            self._save(epoch, train_metrics[self.cm])
            self._log(train_metrics, {}, epoch)
            self.fabric.call('on_epoch', self.cm, self.best_metric, self.best_epoch)
        return train_metrics