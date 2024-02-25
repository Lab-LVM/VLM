import os

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from src.data import create_dataset, create_dataloader
from src.initialize import setup_fabric, ObjectFactory
from src.misc import print_meta_data
from src.utils import resume
from src.utils.registry import create_train_engine

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg.wandb = False
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    with wandb.init(project=cfg.info.project, entity=cfg.info.entity, config=wandb_cfg,
                    settings=wandb.Settings(_disable_stats=True, start_method="thread")):
        cfg.info.id = wandb.run.id
        fabric = setup_fabric(cfg)

        factory = ObjectFactory(cfg, fabric)
        model, tokenizer = factory.create_model()  # model, tokenizer

        train_dataset = create_dataset(cfg.dataset, is_train=True, split=cfg.dataset.train, n_shot=cfg.n_shot)
        loaders = create_dataloader(cfg, train_dataset, is_train=True, fill_last=True)

        optimizer, scheduler, n_epochs = factory.create_optimizer_and_scheduler(model, len(loaders))
        criterion = factory.create_criterion()

        model, optimizer, scheduler, start_epoch = resume(model, optimizer, scheduler, cfg, fabric)
        model, optimizer = fabric.setup(model, optimizer)
        loaders = [fabric.setup_dataloaders(loaders), None]

        cfg = factory.cfg
        fabric.loggers[0].update_config(cfg) if cfg.wandb else None
        print_meta_data(cfg, model, *loaders) if cfg.is_master else None

        # Train
        train_engine = create_train_engine(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler,
                                           (start_epoch, n_epochs))

        train_engine()


if __name__ == "__main__":
    main()
