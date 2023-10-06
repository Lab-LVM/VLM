import logging

from lightning.fabric.utilities.rank_zero import rank_zero_only
from termcolor import colored


class CallBack:
    @staticmethod
    @rank_zero_only
    def on_epoch(tm, best_metric, best_epoch):
        logging.info(colored(f'*** Best {tm}: {best_metric:.5f} {best_epoch} ***', 'blue'))

    @staticmethod
    @rank_zero_only
    def on_train(epoch, update_idx, updates_per_epoch, loss, lr, duration, batch_size):
        logging.info(f'{"Train":>5}: {epoch:>3} [{update_idx:>4d}/{updates_per_epoch - 1}] '
                     f'({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  '
                     f'Loss: {loss.item():#.3g}  '
                     f'LR: {lr:.3e}  '
                     f'TP: {batch_size / duration:>7.2f}/s  '
                     )

    @staticmethod
    @rank_zero_only
    def on_eval(metrics, epoch, num_iter, max_iter):
        log = f'{"Eval":>5}: {epoch:>3}: [{num_iter:>4d}/{max_iter}]  '
        if "ConfusionMatrix" in metrics:
            metrics.pop('ConfusionMatrix')
        for k, v in metrics.items():
            log += f'{k}: {v.item():.4f}  '
        log = log[:-3]

        logging.info(log)
