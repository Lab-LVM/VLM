import os

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig

from src.engine import *
from src.initialize import setup_fabric, ObjectFactory
from src.models import *
from src.utils import dataset2dict, to_list, import_config, move_dir
from src.utils.registry import create_task_engine

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@hydra.main(config_path="configs", config_name="eval_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = import_config(cfg) if cfg.import_cfg else cfg

    cfg.wandb = False
    fabric = setup_fabric(cfg)

    factory = ObjectFactory(cfg, fabric)
    model, tokenizer = factory.create_model()  # model, tokenizer

    df = pd.DataFrame()

    for k, v in dataset2dict(cfg.dataset).items():
        for shot in to_list(cfg.n_shot):
            cfg.dataset = v
            train_dataset = create_dataset(cfg.dataset, split=cfg.dataset.train)
            test_dataset = create_dataset(cfg.dataset, split=cfg.dataset.test)

            engine = create_task_engine(cfg, fabric, model, tokenizer, train_dataset, test_dataset)
            metrics = engine(n_shots=to_list(cfg.n_shot))

            row = dict(Data=test_dataset.name, shot=shot, **metrics)
            print(f'{row}\n')
            df = pd.concat([df, pd.DataFrame(row, index=[0])])

    df.to_csv('result.csv', index=False)

    if cfg.is_master:
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish(quiet=True)

    move_dir(cfg) if cfg.import_cfg else None


if __name__ == "__main__":
    main()
