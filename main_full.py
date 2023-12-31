import os

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig

from src.engine import *
from src.models import *
from src.initialize import setup_fabric, ObjectFactory
from src.misc import print_meta_data
from src.utils import resume, dataset2dict, to_list
from src.utils.registry import create_train_engine

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def check_environment(cfg):
    if 'full' not in cfg.eval_dataset.name:
        print(f'{cfg.eval_dataset.name} is not fully dataset. Changed')
        cfg.eval_dataset.name = 'imagenet_ds_full'
    if cfg.model.forward_backbone == False:
        print(f'Model\'s backbone is not forwarded. Changed')
        cfg.model.forward_backbone = True
    if cfg.dataset.name != 'imagenetraText':
        print(f'Dataset need to fully forwarded.')
        cfg.dataset.name = 'imagenetraText'

    # Model setting
    print("==== Major Setting ====")
    print(f"Language Adapter: {cfg.model.language_adapter}")
    print(f"Visual Adapter: {cfg.model.vision_adapter}")
    print(f"Forward Backbone: {cfg.model.forward_backbone}")
    print(f"Train Dataset: {cfg.dataset.name}")
    print(f"Eval Dataset: {cfg.eval_dataset.name}")
    return cfg


@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg.wandb = False
    cfg = check_environment(cfg)

    fabric = setup_fabric(cfg)

    factory = ObjectFactory(cfg, fabric)
    model, tokenizer = factory.create_model()  # model, tokenizer

    train_dataset = create_dataset(cfg.dataset, is_train=True, split=cfg.dataset.train)
    loaders = create_dataloader(cfg, train_dataset, is_train=True)

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
    #
    # # Eval
    # cfg.train.batch_size = 1024
    # model.eval()
    #
    # df = pd.DataFrame()
    #
    # for k, v in dataset2dict(cfg.eval_dataset).items():
    #     for shot in to_list(cfg.n_shot):
    #         v.name = v.name+'_full'
    #         cfg.dataset = v
    #         train_dataset = create_dataset(cfg.dataset, is_train=True, split=cfg.dataset.train)
    #         test_dataset = create_dataset(cfg.dataset, is_train=False, split=cfg.dataset.test)
    #
    #         engine = OurFullyTaskEngine(cfg, fabric, model, tokenizer, train_dataset, test_dataset)
    #         metrics = engine(n_shots=to_list(cfg.n_shot))
    #
    #         row = dict(Data=test_dataset.name, shot=shot, **metrics)
    #         print(f'{row}\n')
    #         df = pd.concat([df, pd.DataFrame(row, index=[0])])
    #
    # df.to_csv(f'result_{cfg.name}.csv', index=False)
    #
    # if cfg.is_master:
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
