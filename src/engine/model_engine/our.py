import torch

from .. import TaskEngine
from ..feature_engine import ClassificationFeatureEngine
from ..train_engine import TrainEngine
from ...data.dataset import ImageNetRandaugPromptV2, ImageNetRandaugPrompt
from ...utils.loss_function import IndomainOutdomainContrastiveLoss, SupervisedContrastiveLoss
from ...utils.registry import register_task_engine, register_train_engine, register_feature_engine


@register_feature_engine
class OurClassificationFeatureEngine(ClassificationFeatureEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_task_engine
class OurTaskEngine(TaskEngine):
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        feature_engine = OurClassificationFeatureEngine(cfg, fabric, model, tokenizer, train_dataset,
                                                        val_dataset)
        super().__init__(feature_engine)

        self.cfg = cfg
        self.fabric = fabric
        self.device = fabric.device
        self.model = model
        self.val_dataset = val_dataset
        self.logging_interval = cfg.train.log_interval

    @property
    def available_task(self):
        return ['classification']

    def classification(self, **kwargs):
        self.feature_engine.sampling(0)
        self.metric.reset()

        text_classifier = self.feature_engine.build_text_classifier()
        qry_features, qry_labels = self.feature_engine.build_query_set()

        logits = self.model.logit_scale.exp() * qry_features @ text_classifier.mT

        if self.cfg.dataset.name == 'objectnet':
            logits = self.val_dataset.project_logits(logits)

        # Classifier logits
        try:
            classifier_logits = self.model.classifier(qry_features)
            if hasattr(self.val_dataset, 'project_logits'):
                classifier_logits = self.val_dataset.project_logits(classifier_logits)
            logits += classifier_logits
        except:
            pass

        self.metric.update(logits, qry_labels)
        self.metric.prefix = 'simple_adapter_classification'
        return self._output


@register_train_engine
class OurTrainEngine(TrainEngine):
    def __init__(self, cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs):
        super().__init__(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs)
        self.crossentropy = torch.nn.CrossEntropyLoss()

        if hasattr(criterion[0], 'rank'):
            criterion[0].rank = fabric.local_rank
            criterion[0].world_size = fabric.world_size

        if isinstance(criterion[0], IndomainOutdomainContrastiveLoss):
            self.criterion_forward = self.IO_forward
        elif isinstance(criterion[0], SupervisedContrastiveLoss):
            self.criterion_forward = self.SCL_forward
        else:
            self.criterion_forward = self.CLCR_forward

        if isinstance(self.train_loader.dataset, ImageNetRandaugPromptV2):
            self.iterate = self.iterate_ra2
        if isinstance(self.train_loader.dataset, ImageNetRandaugPrompt):
            self.train_loader.dataset.setup_prompt_transform()

    def IO_forward(self, criterion, y, image_feature, text_feature, image_prob=None):
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * torch.mm(image_feature, text_feature.t())
        logits_per_text = logits_per_image.t()
        logits_image_self = logit_scale * torch.mm(image_feature, image_feature.t())
        logits_text_self = logit_scale * torch.mm(text_feature, text_feature.t())

        loss = criterion(logits_per_image, logits_per_text, logits_image_self, logits_text_self, y)
        if image_prob is not None:
            loss = loss + self.crossentropy(image_prob, y) * 0.2

        return loss

    def SCL_forward(self, criterion, y, logits_per_image, logits_per_text, image_prob=None):
        loss = (criterion(logits_per_image, y) + criterion(logits_per_text, y)) / 2
        if image_prob is not None:
            loss = loss + self.crossentropy(image_prob, y) * 0.3
        return loss

    def CLCR_forward(self, criterion, y, logits_per_image, logits_per_text):
        loss = criterion(logits_per_image, logits_per_text)
        return loss

    def iterate(self, model, data, criterion):  # for additional classifier
        x, y, ra_prompt = map(lambda a: a.to(self.device, non_blocking=True), data)

        with self.fabric.autocast():
            outs = model(x, ra_prompt)
            loss = self.criterion_forward(criterion, y, *outs)

        return loss, outs[0], y

    def iterate_ra2(self, model, data, criterion):
        x, ra_x, y, prompt, ra_prompt = map(lambda a: a.to(self.device, non_blocking=True), data)

        x = torch.concat([x, ra_x])
        y = torch.concat([y, y])
        prompt = torch.concat([prompt, ra_prompt])

        with self.fabric.autocast():
            outs = model(x, prompt)
            loss = self.criterion_forward(criterion, y, *outs)

        return loss, outs[0], y

    def __call__(self, *args, **kwargs):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train_loader.sampler.set_epoch(epoch) if self.distributed else None
            self.train_loader.dataset.set_feature(epoch) if hasattr(self.train_loader.dataset, 'set_feature') else None

            train_metrics = self.train(epoch)
            self._distribute_bn()
            self.scheduler.step(epoch + 1)

            self._save(epoch, train_metrics[self.cm])
            self._log(train_metrics, {}, epoch)
            self.fabric.call('on_epoch', self.cm, self.best_metric, self.best_epoch)
