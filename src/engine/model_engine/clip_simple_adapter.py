import torch

from .. import TaskEngine
from ..feature_engine import ClassificationFeatureEngine
from ..train_engine import TrainEngine
from ...data.dataset.imagenet_x import imagenet_a_class_number, imagenet_r_class_number
from ...utils.registry import register_task_engine, register_train_engine, register_feature_engine


@register_feature_engine
class CLIP_SimpleAdapterClassificationFeatureEngine(ClassificationFeatureEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_task_engine
class CLIP_SimpleAdapterTaskEngine(TaskEngine):
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        feature_engine = CLIP_SimpleAdapterClassificationFeatureEngine(cfg, fabric, model, tokenizer, train_dataset,
                                                                       val_dataset)
        super().__init__(feature_engine)

        self.cfg = cfg
        self.fabric = fabric
        self.device = fabric.device
        self.model = model
        self.train_dataset = train_dataset
        self.logging_interval = cfg.train.log_interval

    @property
    def available_task(self):
        return ['classification_zeroshot']

    def classification_zeroshot(self, **kwargs):
        self.feature_engine.sampling(0)
        self.metric.reset()

        text_classifier = self.feature_engine.build_text_classifier()
        qry_features, qry_labels = self.feature_engine.build_query_set()

        logits = self.model.logit_scale.exp() * qry_features @ text_classifier.mT

        # Classifier logits
        # classifier_logits = self.model.classifier(qry_features)
        # if self.cfg.dataset.name == 'imagenet_r':
        #     classifier_logits = classifier_logits[:, imagenet_r_class_number]
        # elif self.cfg.dataset.name == 'imagenet_a':
        #     classifier_logits = classifier_logits[:, imagenet_a_class_number]
        # elif self.cfg.dataset.name == 'objectnet':
        #     classifier_logits = self.train_dataset.to_imageNet_logits(classifier_logits)
        # logits += classifier_logits

        self.metric.update(logits, qry_labels)
        self.metric.prefix = 'clip_zeroshot'
        return self._output


@register_train_engine
class CLIP_SimpleAdapterTrainEngine(TrainEngine):
    def __init__(self, cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs):
        super().__init__(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs)
        self.train_loader.dataset.setup_prompt_transform()
        self.crossentropy = torch.nn.CrossEntropyLoss()

    def iterate(self, model, data, criterion):  # for additional classifier
        x, y, ra_prompt = data

        x = x.to(self.device).to(memory_format=torch.channels_last)
        y = y.to(self.device)
        onehot_y = torch.arange(x.shape[0]).long().to(self.device)
        ra_prompt = self._tokenize(ra_prompt).to(self.device)

        with self.fabric.autocast():
            logits_per_image, logits_per_text, image_prob = model(x, ra_prompt)
            loss = (criterion(logits_per_image, logits_per_text) + self.crossentropy(image_prob, y)) / 2

        return loss, logits_per_image, onehot_y

    def iterate_origin(self, model, data, criterion):
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
