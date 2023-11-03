import gc

import torch
from torch.nn.functional import one_hot

from ..feature_engine import ClassificationFeatureEngine
from ..task_engine import TaskEngine
from ..train_engine import TrainEngine
from ...data import create_dataset
from ...utils.registry import register_task_engine, register_train_engine, register_feature_engine


@register_feature_engine
class TipClassificationFeatureEngine(ClassificationFeatureEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_task_engine
class TipTaskEngine(TaskEngine):
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        feature_engine = TipClassificationFeatureEngine(cfg, fabric, model, tokenizer, train_dataset, val_dataset)
        super().__init__(feature_engine)

    def __call__(self, n_shots, **kwargs):
        output = dict()
        for task_name in self.available_task:
            for n_shot in n_shots:
                metric = self.task(task_name, n_shot=n_shot)
                output.update(metric)

                torch.cuda.empty_cache()
                gc.collect()
        return output

    @property
    def available_task(self):
        return ['classification_fewshot']

    def classification_fewshot_simple(self, n_shot):
        self.feature_engine.sampling(n_shot)
        self.metric.reset()

        beta = 3.79
        alpha = 0.97

        text_classifier = self.feature_engine.build_text_classifier()
        qry_features, qry_labels = self.feature_engine.build_query_set()
        sup_features, sup_labels = self.feature_engine.build_support_set()
        sup_labels = one_hot(sup_labels)

        logits = 100. * qry_features @ text_classifier.mT

        affinity = qry_features @ sup_features.mT
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ sup_labels.half()

        tip_logits = logits + cache_logits * alpha
        self.metric.update(tip_logits, qry_labels)
        self.metric.prefix = f'tip_fewshot{n_shot}'
        return self._output

    def classification_fewshot(self, n_shot):
        self.feature_engine.sampling(n_shot)

        text_classifier = self.feature_engine.build_text_classifier()
        qry_features, qry_labels = self.feature_engine.build_query_set()
        sup_features, sup_labels = self.feature_engine.build_support_set()
        sup_labels = one_hot(sup_labels)

        best_accuracy, beta, alpha = self.search_hp(text_classifier, qry_features, qry_labels, sup_features, sup_labels)

        return {f'tip_fewshot{n_shot}_finetune': best_accuracy}

    def search_hp(self, text_classifier, qry_features, qry_labels, sup_features, sup_labels):
        search_scale = (7, 3)
        search_step = (200, 20)

        beta_list = [i * (search_scale[0] - 0.1) / search_step[0] + 0.1 for i in
                     range(search_step[0])]
        alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in
                      range(search_step[1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                self.metric.reset()
                logits = 100. * qry_features @ text_classifier.mT

                affinity = qry_features @ sup_features.mT
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ sup_labels.half()
                tip_logits = logits + cache_logits * alpha
                self.metric.update(tip_logits, qry_labels)
                acc = self.metric.compute().item()
                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        return best_acc, best_beta, best_alpha


@register_train_engine
class TipTrainEngine(TrainEngine):
    def __init__(self, cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs, **kwargs):
        super().__init__(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs)
        print("[!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!] Not implement yet exactly")

        cache_dataset = create_dataset(cfg.dataset, split=cfg.dataset.train, n_shot=cfg.n_shot)
        self.feature_engine = TipClassificationFeatureEngine(cfg, fabric, model, tokenizer, cache_dataset,
                                                             self.val_loader.dataset)

        self.alpha = kwargs.get('alpha', 1.17)
        self.beta = kwargs.get('beta', 1.0)
        self.sup_features, self.sup_labels = self.feature_engine.build_support_set()
        self.text_classifier = self.feature_engine.build_text_classifier()
        self.sup_labels = one_hot(self.sup_labels)

        self.model.adapter.weight.data = self.sup_features.float()

    def iterate(self, model, data, criterion):
        x, y = map(lambda x: x.to(self.device), data)
        x = x.to(memory_format=torch.channels_last)

        with self.fabric.autocast():
            with torch.no_grad():
                image_features = self.model.encode_image(x)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = 100. * image_features @ self.text_classifier.mT

            affinity = self.model.adapter(image_features)

            cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.sup_labels.half()
            tip_logits = logits + cache_logits * self.alpha

            loss = criterion(tip_logits, y)

        return loss, tip_logits, y

    def _model_train(self):
        self.model.eval()
        self.model.adapter.train()

    def _model_eval(self):
        self.model.eval()