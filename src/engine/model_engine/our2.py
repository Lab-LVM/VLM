import copy
import gc

import pandas as pd
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from .. import TaskEngine
from ..feature_engine import ClassificationFeatureEngine
from ..train_engine import TrainEngine
from ...data import create_dataset
from ...data.dataset import ImageNetRandaugPrompt, ImageNetRandaugPromptText
from ...data.dataset import ObjectNet
from ...data.dataset.imagenet_text import ImageNetSimplePromptText, ImageNetRandaugPromptOriginalText
from ...utils import dataset2dict, to_list
from ...utils.loss_function import IndomainOutdomainContrastiveLoss, SupervisedContrastiveLoss, \
    SupervisedContrastiveLossMultiProcessing, CLIPLoss, SoftCLIPLoss
from ...utils.registry import register_task_engine, register_train_engine, register_feature_engine, create_task_engine


@register_feature_engine
class Our2ClassificationFeatureEngine(ClassificationFeatureEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



@register_task_engine
class Our2TaskEngine(TaskEngine):
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        feature_engine = Our2ClassificationFeatureEngine(cfg, fabric, model, tokenizer, train_dataset,
                                                        val_dataset)
        super().__init__(feature_engine)

        self.cfg = cfg
        self.fabric = fabric
        self.device = fabric.device
        self.model = model
        self.val_dataset = val_dataset
        self.logging_interval = cfg.train.log_interval

        self.metric = Accuracy('multiclass', num_classes=cfg.dataset.num_classes).cuda()
        self.model.eval()

    @property
    def available_task(self):
        return ['classification']

    def classification(self, **kwargs):
        self.feature_engine.sampling(0)
        self.metric.reset()

        text_classifier = self.feature_engine.build_text_classifier()
        qry_features, qry_labels = self.feature_engine.build_query_set()

        logits = self.model.logit_scale.exp() * qry_features @ text_classifier.mT

        if hasattr(self.val_dataset, 'project_logits'):
            logits = self.val_dataset.project_logits(logits)
        print(logits.shape, qry_labels.shape)
        self.metric.update(logits, qry_labels)
        self.metric.prefix = 'simple_adapter_classification'
        return self._output


@register_train_engine
class Our2TrainEngine(TrainEngine):
    def __init__(self, cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs):
        super().__init__(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs)
        self.model.cuda()
        self.train_loader.dataset.setup_prompt_transform() if hasattr(self.train_loader.dataset,
                                                                      'setup_prompt_transform') else None

        if hasattr(criterion[0], 'rank'):
            criterion[0].rank = fabric.local_rank
            criterion[0].world_size = fabric.world_size

        if isinstance(criterion[0], IndomainOutdomainContrastiveLoss):
            self.criterion_forward = self.IOL_forward
        elif isinstance(criterion[0], SupervisedContrastiveLoss):
            self.criterion_forward = self.SCL_forward
        elif isinstance(criterion[0], (SupervisedContrastiveLossMultiProcessing, CLIPLoss, SoftCLIPLoss)):
            self.criterion_forward = self.SCLM_forward
        else:
            self.criterion_forward = self.CLCR_forward

        if isinstance(self.train_loader.dataset, ImageNetRandaugPrompt):
            self.train_loader.dataset.setup_prompt_transform()

    def IOL_forward(self, criterion, y, image_feature, text_feature):
        logit_scale = self.model.logit_scale.exp()
        loss = criterion(image_feature, text_feature, y, logit_scale)
        return loss

    def SCLM_forward(self, criterion, y, image_feature, text_feature, image_prob=None):
        loss = criterion(image_feature, text_feature, y, self.model.logit_scale.exp())
        return loss

    def SCL_forward(self, criterion, y, logits_per_image, logits_per_text, image_prob=None):
        loss = (criterion(logits_per_image, y) + criterion(logits_per_text, y)) / 2
        return loss

    def CLCR_forward(self, criterion, y, logits_per_image, logits_per_text):
        loss = criterion(logits_per_image, logits_per_text)
        return loss

    def iterate(self, model, data, criterion):
        x, ra_x, y, prompt, ra_prompt = data
        prompt = self.tokenizer(prompt, padding='max_length', return_attention_mask=False, return_tensors='pt')[
            'input_ids']
        ra_prompt = self.tokenizer(ra_prompt, padding='max_length', return_attention_mask=False, return_tensors='pt')[
            'input_ids']

        x = torch.concat([x, ra_x]).cuda() #.to(self.device, non_blocking=True)
        y = torch.concat([y, y]).cuda() #.to(self.device, non_blocking=True)
        prompt = torch.concat([prompt, ra_prompt]).cuda() #.to(self.device, non_blocking=True)

        with self.fabric.autocast():
            outs = model(x, prompt)
            loss = self.criterion_forward(criterion, y, *outs)

        return loss, outs[0], y

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            df = pd.DataFrame()
            cfg = copy.deepcopy(self.cfg)
            ds_backbone = cfg.model.backbone.split('-')[-1]
            for k, v in dataset2dict(cfg.eval_dataset).items():
                cfg.dataset = v
                test_dataset = create_dataset(cfg.dataset, is_train=False, split=cfg.dataset.test, backbone=ds_backbone)

                engine = create_task_engine(cfg, self.fabric, self.model, self.tokenizer, test_dataset, test_dataset)
                metrics = engine(n_shots=to_list(cfg.n_shot))

                row = dict(Data=test_dataset.name, Acc=float(metrics['simple_adapter_classification']))
                df = pd.concat([df, pd.DataFrame(row, index=[0])])
            df = pd.concat([df, pd.DataFrame({'Data': 'Mean', 'Acc': df['Acc'].mean()}, index=[0])])
        del test_dataset
        torch.cuda.empty_cache()
        gc.collect()
        return df

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

        if kwargs.get('pass_eval', None):
            return None
        return self.eval()
