import gc
from abc import ABC, abstractmethod

import torch
from termcolor import colored
from torch.nn.functional import one_hot
from torchmetrics import Accuracy

from src.utils.registry import register_engine
from .feature_engine import FeatureEngine, CLIPClassificationFeatureEngine, TIPClassificationFeatureEngine


class ModelEngine(ABC):
    def __init__(self, feature_engine: FeatureEngine):
        self.feature_engine = feature_engine
        self.metric = Accuracy('multiclass', num_classes=feature_engine.num_class).to(feature_engine.device)

    @abstractmethod
    def task(self, *args, **kwargs):
        pass

    def warning(self, text):
        print(f'{colored(f"Warning:[{self.__class__.__name__}]", "red")} {text}')

    @property
    def _output(self):
        return {self.metric.prefix: self.metric.compute().item()}


@register_engine
class CLIPEngine(ModelEngine):
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        feature_engine = CLIPClassificationFeatureEngine(cfg, fabric, model, tokenizer, train_dataset, val_dataset)
        super().__init__(feature_engine)

        self.available_task = ['classification_zeroshot', 'classification_linear_prob']

    def __call__(self, *args, **kwargs):
        output = dict()
        for task_name in self.available_task:
            metric = self.task(task_name)
            output.update(metric)

            torch.cuda.empty_cache()
            gc.collect()

        return output

    def task(self, task_name):
        if 'classification' in task_name:
            if 'zeroshot' in task_name:
                return self.classification_zeroshot()
            else:
                self.warning(f'{task_name} is not implemented.')
                return self.classification_linear_prob()
        else:
            return self.warning(f'{task_name} is not implemented.')

    def classification_zeroshot(self):
        self.feature_engine.sampling(0)
        self.metric.reset()

        text_classifier = self.feature_engine.build_text_classifier()
        qry_features, qry_labels = self.feature_engine.build_query_set()

        logits = 100. * qry_features @ text_classifier.mT

        self.metric.update(logits, qry_labels)
        self.metric.prefix = 'clip_zeroshot'
        return self._output

    def classification_linear_prob(self):
        return {}


@register_engine
class TipEngine(ModelEngine):
    def __init__(self, cfg, fabric, model, tokenizer, train_dataset, val_dataset):
        feature_engine = TIPClassificationFeatureEngine(cfg, fabric, model, tokenizer, train_dataset, val_dataset)
        super().__init__(feature_engine)

        self.available_task = ['classification_fewshot', 'classification_fewshot_finetune']

    def __call__(self, n_shots, *args, **kwargs):
        output = dict()
        for task_name in self.available_task:
            for n_shot in n_shots:
                if n_shot == 0:
                    continue
                metric = self.task(task_name, n_shot)
                output.update(metric)

                torch.cuda.empty_cache()
                gc.collect()

        return output

    def task(self, task_name, n_shot):
        if 'classification' in task_name:
            if 'finetune' in task_name:
                self.warning(f'{task_name} is not implemented.')
                return self.classification_fewshot_finetune(n_shot)
            else:
                return self.classification_fewshot(n_shot)
        else:
            return self.warning(f'{task_name} is not implemented.')

    def classification_fewshot(self, n_shot, beta=1, alpha=1.17):
        self.feature_engine.sampling(n_shot)
        self.metric.reset()

        text_classifier = self.feature_engine.build_text_classifier()
        qry_features, qry_labels = self.feature_engine.build_query_set()
        sup_features, sup_labels = self.feature_engine.build_support_set()
        sup_labels = one_hot(sup_labels)

        # beta, alpha = self.search_hp(text_classifier, qry_features, qry_labels, sup_features, sup_labels)
        beta = 3.79
        alpha = 0.97

        logits = 100. * qry_features @ text_classifier.mT

        affinity = qry_features @ sup_features.mT
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ sup_labels.half()

        tip_logits = logits + cache_logits * alpha
        self.metric.update(tip_logits, qry_labels)
        self.metric.prefix = f'tip_fewshot{n_shot}'
        return self._output

    def classification_fewshot_finetune(self, n_shot):
        return {}

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
                    print(
                        "New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

        return best_beta, best_alpha
