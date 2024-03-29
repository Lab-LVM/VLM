import gc
import os

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig

from src.engines import *
from src.data import create_dataset, create_dataloader
from src.initialize import setup_fabric, ObjectFactory
from src.misc import print_meta_data
from src.utils import resume
from src.utils.registry import create_train_engine, create_task_engine

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    fabric = setup_fabric(cfg)

    factory = ObjectFactory(cfg, fabric)
    model, tokenizer = factory.create_model()  # model, tokenizer

    train_dataset = create_dataset(cfg.dataset, is_train=True, split=cfg.dataset.train, n_shot=cfg.n_shot)
    loaders = create_dataloader(cfg, train_dataset, is_train=True)

    optimizer, scheduler, n_epochs = factory.create_optimizer_and_scheduler(model, len(loaders))
    criterion = factory.create_criterion()

    model, optimizer, scheduler, start_epoch = resume(model, optimizer, scheduler, cfg, fabric)
    model, optimizer = fabric.setup(model, optimizer)
    loaders = [fabric.setup_dataloaders(loaders), None]

    cfg = factory.cfg
    fabric.loggers[0].update_config(cfg) if cfg.wandb else None
    print_meta_data(cfg, model, *loaders) if cfg.is_master else None

    train_engine = create_train_engine(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler,
                                       (start_epoch, n_epochs))

    train_engine()

    # Evaluation
    test_dataset = create_dataset(cfg.dataset, is_train=False, split=cfg.dataset.test, n_shot=cfg.n_shot)

    engine = create_task_engine(cfg, fabric, model, tokenizer, train_dataset, test_dataset)
    metrics = engine(n_shot=cfg.n_shot)
    row = dict(ExpName=cfg.name, Data=test_dataset.name, **metrics)

    with fabric.rank_zero_first():
        print(f'{row}\n')
        df = pd.DataFrame(row, index=[0])
        df.to_csv(f'{cfg.name}_result.csv', index=False)

    if cfg.is_master:
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
