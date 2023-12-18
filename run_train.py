import gc
import os

import hydra
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data import create_dataloader
from src.engine import *
from src.initialize import setup_fabric, ObjectFactory
from src.misc import print_meta_data
from src.models import *
from src.utils import resume
from src.utils.registry import create_train_engine


@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg.wandb = False
    fabric = setup_fabric(cfg)

    factory = ObjectFactory(cfg, fabric)
    model, tokenizer = factory.create_model()  # model, tokenizer

    train_dataset = create_dataset(cfg.dataset, split=cfg.dataset.train, n_shot=cfg.n_shot)
    loaders = [DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers), None]

    optimizer, scheduler, n_epochs = factory.create_optimizer_and_scheduler(model, len(loaders[0]))
    criterion = factory.create_criterion()

    model, optimizer, scheduler, start_epoch = resume(model, optimizer, scheduler, cfg, fabric)
    model, optimizer = fabric.setup(model, optimizer)

    cfg = factory.cfg
    fabric.loggers[0].update_config(cfg) if cfg.wandb else None
    print_meta_data(cfg, model, *loaders) if cfg.is_master else None

    train_engine = create_train_engine(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler,
                                       (start_epoch, n_epochs))

    train_engine()

    if cfg.is_master:
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
