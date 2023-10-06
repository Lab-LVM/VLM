import os

import pandas as pd
import wandb
from omegaconf import DictConfig

from src.engine import *
from src.initialize import setup_fabric, ObjectFactory
from src.utils.registry import create_engine
from src.utils.utils import dataset2dict, to_list
from src.models import *


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg.wandb = False
    fabric = setup_fabric(cfg)

    factory = ObjectFactory(cfg, fabric)
    model, tokenizer = factory.create_model()  # model, tokenizer

    df = pd.DataFrame()

    for k, v in dataset2dict(cfg.dataset).items():
        cfg.dataset = v
        train_dataset = create_dataset(cfg.dataset, split=cfg.dataset.train)
        val_dataset = create_dataset(cfg.dataset, split=cfg.dataset.valid)

        engine = create_engine(cfg, fabric, model, tokenizer, train_dataset, val_dataset)
        metrics = engine(n_shots=to_list(cfg.n_shot))

        row = dict(Data=k, **metrics)
        print(f'{row}\n')
        df = pd.concat([df, pd.DataFrame(row, index=[0])])

    print(df)
    df.to_csv('result.csv', index=False)

    if cfg.is_master:
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
