import shutil
from time import time

import datasets
import torch
import torchmetrics
from torchmetrics import MeanMetric

datasets.disable_progress_bar()


class TrainEngine:
    def __init__(self, cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs):
        self.local_rank = fabric.local_rank
        self.world_size = fabric.world_size
        self.device = fabric.device
        self.fabric = fabric
        self.cfg = cfg

        self.distributed = True if fabric.world_size > 1 else False
        self.dist_bn = cfg.train.dist_bn

        self.start_epoch, self.num_epochs = epochs
        self.logging_interval = cfg.train.log_interval
        self.model_name = cfg.model.model_name
        self.num_classes = cfg.dataset.num_classes
        self.sample_size = cfg.train.batch_size * cfg.train.optimizer.grad_accumulation * self.world_size

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_accumulation = cfg.train.optimizer.grad_accumulation
        self.train_criterion, self.val_criterion = criterion
        self.train_loader, self.val_loader = loaders

        self.cm = cfg.train.criteria_metric
        self.decreasing = cfg.train.criteria_decreasing
        self.losses = MeanMetric().to(self.device)
        self.metric_fn = self._init_metrics(cfg.dataset.task, cfg.train.eval_metrics,
                                            0.5, self.num_classes, self.num_classes, 'macro')
        self.best_metric = self.best_epoch = 0 if not self.decreasing else float('inf')

    def __call__(self, *args, **kwargs):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train_loader.sampler.set_epoch(epoch) if self.distributed else None

            train_metrics = self.train(epoch)
            self._distribute_bn()
            eval_metrics = self.eval(epoch)

            self.scheduler.step(epoch + 1)

            self._save(epoch, eval_metrics[self.cm])
            self._log(train_metrics, eval_metrics, epoch)
            self.fabric.call('on_epoch', self.cm, self.best_metric, self.best_epoch)

    def iterate(self, model, data, criterion):
        x, y = map(lambda a: a.to(self.device), data)
        x = x.to(memory_format=torch.channels_last)

        with self.fabric.autocast():
            prob = model(x)
            loss = criterion(prob, y)

        return loss, prob, y

    def train(self, epoch):
        self._reset_metric()

        total_len = len(self.train_loader)
        accum_steps = self.grad_accumulation
        num_updates = epoch * (update_len := (total_len + accum_steps - 1) // accum_steps)
        total_len = total_len - 1

        self.model.train()
        self.optimizer.zero_grad()
        start = time()
        for i, data in enumerate(self.train_loader):
            is_accumulating = (i + 1) % accum_steps != 0
            update_idx = i // accum_steps

            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                loss, prob, target = self.iterate(self.model, data, self.train_criterion)
                self.fabric.backward(loss)

            self.losses.update(loss)

            if is_accumulating:
                continue

            self.optimizer.step()
            self.optimizer.zero_grad()

            tnow = time()
            duration, start = tnow - start, tnow
            num_updates += 1
            if update_idx % self.logging_interval == 0 or i == total_len:
                lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                self.fabric.call('on_train', epoch, update_idx, update_len, loss, lr, duration, self.sample_size)

            self.scheduler.step_update(num_updates=num_updates, metric=self.losses.compute())

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return {'loss': self.losses.compute().item()}

    @torch.no_grad()
    def eval(self, epoch):
        self._reset_metric()
        total = len(self.val_loader) - 1

        self._model_eval()
        for i, data in enumerate(self.val_loader):
            loss, prob, target = self.iterate(self.model, data, self.val_criterion)
            self._update_metric(loss, prob, target)

            if i % self.logging_interval == 0 or i == total:
                self.fabric.call('on_eval', self._metrics(), epoch, i, total)

        return self._metrics()

    def _model_train(self):
        self.model.train()

    def _model_eval(self):
        self.model.eval()

    def _distribute_bn(self):
        # ensure every node has the same running bn stats
        reduce = self.dist_bn == 'reduce'
        if self.distributed and self.dist_bn in ('broadcast', 'reduce'):
            for bn_name, bn_buf in self.model.named_buffers(recurse=True):
                if ('running_mean' in bn_name) or ('running_var' in bn_name):
                    self.fabric.all_reduce(bn_buf) if reduce else self.fabric.broadcast(bn_buf, 0)

    def _update_metric(self, loss, prob, target):
        self.losses.update(loss.item())
        for fn in self.metric_fn.values():
            fn.update(prob, target)

    def _reset_metric(self):
        self.losses.reset()
        for fn in self.metric_fn.values():
            fn.reset()

    def _metrics(self):
        result = {'loss': self.losses.compute(), **self.metric_fn.compute()}
        return result

    def _save(self, epoch, criterion_metric=None):
        save_path = 'latest.ckpt'
        save_state = {
            'epoch': epoch,
            'arch': self.model_name,
            'state_dict': self.model,
            'optimizer': self.optimizer,
            'cfg': self.cfg,
        }
        if criterion_metric is not None:
            save_state[self.cm] = criterion_metric
        self.fabric.save(save_path, save_state)

        is_best = (self.decreasing and criterion_metric < self.best_metric) or (
                not self.decreasing and criterion_metric > self.best_metric)
        if is_best:
            self.best_metric, self.best_epoch = criterion_metric, epoch
            if self.fabric.local_rank == 0:
                shutil.copy(save_path, 'best.ckpt')

    def _init_metrics(self, task, eval_metrics, threshold, num_class, num_label, average, top_k=1):
        if eval_metrics == 'all':
            eval_metrics = ['Top1', 'Top5', 'F1Score', 'Specificity', 'Recall', 'Precision', 'AUROC',
                            'ConfusionMatrix']

        metric_fn = dict()
        for metric in eval_metrics:
            if metric in ['Top1', 'Top5', 'Accuracy']:
                metric_fn[metric] = torchmetrics.__dict__['Accuracy'](task=task, threshold=threshold, average='micro',
                                                                      num_classes=num_class, num_labels=num_label,
                                                                      top_k=int(metric[-1]))
            elif metric in ['AUROC']:
                metric_fn[metric] = torchmetrics.__dict__['AUROC'](task=task, num_classes=num_class,
                                                                   num_labels=num_label, average='macro')
            elif metric in ['ConfusionMatrix']:
                metric_fn[metric] = torchmetrics.__dict__['ConfusionMatrix'](task=task, num_classes=num_class,
                                                                             num_labels=num_label)
            else:
                metric_fn[metric] = torchmetrics.__dict__[metric](task=task, threshold=threshold, num_classes=num_class,
                                                                  average=average, num_labels=num_label, top_k=top_k)

        metric_fn = torchmetrics.MetricCollection(metric_fn).to(self.device)
        return metric_fn

    def _log(self, train_metrics, eval_metrics, epoch):
        eval_metrics.update({f'Best_{self.cm}': self.best_metric})
        metrics = {**self._add_prefix(train_metrics, 'train'), **self._add_prefix(eval_metrics, 'eval')}
        self.fabric.log_dict(metrics, epoch)

    @staticmethod
    def _add_prefix(metrics, prefix, separator='_'):
        if not prefix:
            return metrics
        return {f"{prefix}{separator}{k}": v for k, v in metrics.items()}

    def _tokenize(self, text):
        dataset = datasets.Dataset.from_dict({'text': text})

        text_embedding = dataset.map(
            lambda item: self.tokenizer(item['text'], padding='max_length', return_attention_mask=False),
            remove_columns=['text'], batched=True).with_format('pt', device=self.device)

        return text_embedding['input_ids']
