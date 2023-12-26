from copy import copy

import torch
from termcolor import colored
from timm.loss import BinaryCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from torch import nn

from src.utils.loss_function import CLIPLoss, CoCaLoss, SupervisedContrastiveLoss, IndomainOutdomainContrastiveLoss, \
    SupervisedContrastiveLossMultiProcessing
from src.utils.registry import create_model
from src.utils.utils import filter_grad, EmptyScheduler


class ObjectFactory:
    def __init__(self, cfg, fabric):
        self.cfg = cfg
        self.train = cfg.train
        self.optim = cfg.train.optimizer
        self.scheduler = cfg.train.scheduler
        self.dataset = cfg.dataset
        self.model = cfg.model
        self.fabric = fabric

        self.device = fabric.device
        self.checkpoint = cfg.checkpoint

    def create_model(self):
        tokenizer = create_model(self.model.tokenizer)

        model = create_model(
            **self.model,
            in_chans=self.dataset.in_channels,
            num_classes=self.dataset.num_classes,
        )
        self.load_checkpoint(model)
        model.to(self.device)

        if self.train.channels_last:
            model = model.to(memory_format=torch.channels_last)

        if self.cfg.distributed and self.train.sync_bn:
            self.train.dist_bn = ''
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        return model, tokenizer

    def load_checkpoint(self, model):
        if self.checkpoint is not None:
            import os
            import hydra
            state_dict = self.fabric.load(os.path.join(hydra.utils.get_original_cwd(), self.checkpoint))['state_dict']
            model.load_state_dict(state_dict, strict=True)

    def create_optimizer_and_scheduler(self, model, iter_per_epoch):
        self.cfg.train.iter_per_epoch = iter_per_epoch
        self.train.iter_per_epoch = iter_per_epoch
        # self.check_total_batch_size()

        if self.optim.opt == 'lbfgs':
            lbfgs_kwargs = copy(self.optim)
            lbfgs_kwargs.__delattr__('opt')
            lbfgs_kwargs.__delattr__('grad_accumulation')
            optimizer = torch.optim.LBFGS(filter_grad(model), **lbfgs_kwargs)
        else:
            optimizer = create_optimizer_v2(filter_grad(model), **optimizer_kwargs(cfg=self.optim))

        if self.scheduler.sched is not None:
            updates_per_epoch = \
                (iter_per_epoch + self.optim.grad_accumulation - 1) // self.optim.grad_accumulation

            scheduler, num_epochs = create_scheduler_v2(
                optimizer,
                **scheduler_kwargs(self.scheduler),
                updates_per_epoch=updates_per_epoch,
            )
        else:
            scheduler = EmptyScheduler()
            num_epochs = self.train.epochs

        if self.optim.get('clip_grad', None) is not None:
            self.fabric.clip_gradients(model, optimizer, self.optim.clip_grad)
        return optimizer, scheduler, num_epochs

    def create_criterion(self):
        validate_loss_fn = None
        if self.train.criterion == 'CLIPLoss':
            train_loss_fn = validate_loss_fn = CLIPLoss()

        elif self.train.criterion == 'CoCaLoss':
            train_loss_fn = validate_loss_fn = CoCaLoss(1.0, 1.0)

        elif self.train.criterion == 'SCL':
            train_loss_fn = validate_loss_fn = SupervisedContrastiveLoss()

        elif self.train.criterion == 'IOL':
            train_loss_fn = validate_loss_fn = IndomainOutdomainContrastiveLoss()  # contextual

        elif self.train.criterion == 'SCLM':
            train_loss_fn = validate_loss_fn = SupervisedContrastiveLossMultiProcessing()

        elif self.dataset.augmentation.cutmix > 0 or self.dataset.augmentation.mixup > 0:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if self.train.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=self.train.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()

        elif self.dataset.augmentation.smoothing > 0:
            if self.train.bce_loss:
                train_loss_fn = BinaryCrossEntropy(smoothing=self.dataset.augmentation.smoothing,
                                                   target_threshold=self.train.bce_target_thresh)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self.dataset.augmentation.smoothing)

        else:
            train_loss_fn = nn.CrossEntropyLoss()

        train_loss_fn = train_loss_fn
        validate_loss_fn = validate_loss_fn if validate_loss_fn is not None else nn.CrossEntropyLoss()

        self.cfg.train.criterion = train_loss_fn.__class__.__name__
        return train_loss_fn, validate_loss_fn

    def check_total_batch_size(self):
        total_batch = self.train.total_batch
        grad_accum = self.optim.grad_accumulation
        batch_size = self.train.batch_size

        if total_batch < self.cfg.world_size * batch_size:
            batch_size = total_batch // self.cfg.world_size
            self.fabric.print(colored(
                f'[WARNING] Batch size({self.train.batch_size}) is too larger than total batch({total_batch}). Batch size will be set to {batch_size}',
                'red'))

        self.train.optimizer.grad_accumulation = total_batch // (self.cfg.world_size * batch_size)
        self.train.batch_size = batch_size
        self.train.total_batch = total_batch
        self.optim.grad_accumulation = self.train.optimizer.grad_accumulation

        if self.train.batch_size % grad_accum != 0:
            self.fabric.print(f'{colored("[WARNING]", "red")} '
                              f'The batch size({self.train.batch_size}) does not divided grad accumulation size({grad_accum})')
