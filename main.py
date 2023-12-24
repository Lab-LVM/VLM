import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig

from src.engine import *
from src.initialize import setup_fabric, ObjectFactory
from src.misc import print_meta_data
from src.models import *
from src.utils import resume, dataset2dict, to_list
from src.utils.registry import create_train_engine, create_task_engine


@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg.wandb = False
    fabric = setup_fabric(cfg)

    factory = ObjectFactory(cfg, fabric)
    model, tokenizer = factory.create_model()  # model, tokenizer

    train_dataset = create_dataset(cfg.dataset, split=cfg.dataset.train, n_shot=cfg.n_shot)
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

    # Eval
    del model
    cfg.model.eval = True
    cfg.train.batch_size = 1024
    factory = ObjectFactory(cfg, fabric)
    model, _ = factory.create_model()  # model, tokenizer
    state_dict = fabric.load('best.ckpt')['state_dict']
    model.load_state_dict(state_dict, strict=True)

    df = pd.DataFrame()

    for k, v in dataset2dict(cfg.eval_dataset).items():
        for shot in to_list(cfg.n_shot):
            cfg.dataset = v
            train_dataset = create_dataset(cfg.dataset, split=cfg.dataset.train)
            test_dataset = create_dataset(cfg.dataset, split=cfg.dataset.test)

            engine = create_task_engine(cfg, fabric, model, tokenizer, train_dataset, test_dataset)
            metrics = engine(n_shots=to_list(cfg.n_shot))

            row = dict(Data=test_dataset.name, shot=shot, **metrics)
            print(f'{row}\n')
            df = pd.concat([df, pd.DataFrame(row, index=[0])])

    df.to_csv(f'result_{cfg.name}.csv', index=False)

    if cfg.is_master:
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()