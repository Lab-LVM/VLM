import gc
import os

import torch
import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig

from src.engines import *
from src.data import create_dataset
from src.initialize import setup_fabric, ObjectFactory
from src.utils import dataset2dict, to_list
from src.utils.registry import create_task_engine

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@hydra.main(config_path="configs", config_name="eval_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    fabric = setup_fabric(cfg)

    factory = ObjectFactory(cfg, fabric)
    model, tokenizer = factory.create_model()  # model, tokenizer
    model = fabric.setup_module(model)

    df = pd.DataFrame()

    for k, v in dataset2dict(cfg.dataset).items():
        for n_shot in to_list(cfg.n_shot):
            cfg.dataset = v
            train_dataset = create_dataset(cfg.dataset, is_train=True, split=cfg.dataset.train, n_shot=n_shot)
            test_dataset = create_dataset(cfg.dataset, is_train=False, split=cfg.dataset.test)

            engine = create_task_engine(cfg, fabric, model, tokenizer, train_dataset, test_dataset)
            metrics = engine(n_shot=n_shot)

            row = dict(ExpName=cfg.name, Data=test_dataset.name, **metrics)
            print(f'{row}\n')
            df = pd.concat([df, pd.DataFrame(row, index=[0])])

    df.to_csv(f'{cfg.name}_result.csv', index=False)

    if cfg.is_master:
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
