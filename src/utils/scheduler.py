import numpy as np


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


class CosineLR:
    def __init__(self, optimizer, base_lrs, warmup_length, steps):
        self.optimizer = optimizer
        self.base_lrs = base_lrs
        self.warmup_length = warmup_length
        self.steps = steps
        if not isinstance(self.base_lrs, list):
            self.base_lrs = [self.base_lrs for _ in self.optimizer.param_groups]
        assert len(self.base_lrs) == len(self.optimizer.param_groups)

    def _lr_adjust(self, step):
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if step < self.warmup_length:
                lr = _warmup_lr(base_lr, self.warmup_length, step)
            else:
                e = step - self.warmup_length
                es = self.steps - self.warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    def step_update(self, num_updates, **kwargs):
        return self._lr_adjust(num_updates)

    def step(*args, **kwargs):
        pass
