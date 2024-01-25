import os

import hydra
import wandb
from omegaconf import DictConfig

from src.engine import *
from src.models import *
from src.initialize import setup_fabric, ObjectFactory
from src.misc import print_meta_data
from src.utils import resume, dataset2dict, to_list

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def check_environment(cfg):
    if cfg.model.forward_backbone == False:
        print(f'Model\'s backbone is not forwarded. Changed')
        cfg.model.forward_backbone = True

    # Model setting
    if cfg.is_master:
        print("==== Major Setting ====")
        print(f"Language Adapter: {getattr(cfg.model, 'language_adapter', None)}")
        print(f"Visual Adapter: {getattr(cfg.model, 'vision_adapter', None)}")
        print(f"Forward Backbone: {getattr(cfg.model, 'forward_backbone', None)}")
        print(f"Return Feature: {getattr(cfg.model, 'return_feature', None)}")
        print(f"Train Dataset: {cfg.dataset.name}")
        print(f"Eval Dataset: {cfg.eval_dataset.name}")
    return cfg


@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg.wandb = False
    fabric = setup_fabric(cfg)
    cfg = check_environment(cfg)

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
    train_engine = Our2TrainEngineForDistributionShift(cfg, fabric, model, tokenizer, loaders, criterion, optimizer,
                                                       scheduler, (start_epoch, n_epochs))
    train_engine()

    # Eval
    cfg.train.batch_size = 512
    cfg.train.num_workers=4
    model.eval()

    df = pd.DataFrame()

    for k, v in dataset2dict(cfg.eval_dataset).items():
        cfg.dataset = v
        train_dataset = create_dataset(cfg.dataset, is_train=True, split=cfg.dataset.train)
        test_dataset = create_dataset(cfg.dataset, is_train=False, split=cfg.dataset.test)

        engine = create_task_engine(cfg, fabric, model, tokenizer, train_dataset, test_dataset)
        metrics = engine(n_shots=to_list(cfg.n_shot))

        row = dict(Data=test_dataset.name, shot=0, **metrics)
        if fabric.is_global_zero:
            print(f'{row}\n')
        df = pd.concat([df, pd.DataFrame(row, index=[0])])

    if fabric.is_global_zero:
        df.to_csv(f'result_{cfg.name}.csv', index=False)
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
