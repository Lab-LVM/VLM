import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from src.engine import *
from src.models import *
from src.initialize import setup_fabric, ObjectFactory
from src.misc import print_meta_data
from src.utils import resume, dataset2dict, to_list
from src.utils.registry import create_train_engine, create_task_engine


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

        ds_backbone = cfg.model.backbone.split('-')[-1]
        train_dataset = create_dataset(cfg.dataset, backbone=ds_backbone, is_train=False, split=cfg.dataset.train)
        loaders = create_dataloader(cfg, train_dataset, is_train=True)

        optimizer, scheduler, n_epochs = factory.create_optimizer_and_scheduler(model, len(loaders))
        criterion = factory.create_criterion()

        model, optimizer, scheduler, start_epoch = resume(model, optimizer, scheduler, cfg, fabric)
        model, optimizer = fabric.setup(model, optimizer)
        loaders = [fabric.setup_dataloaders(loaders), None]

        cfg = factory.cfg
        fabric.loggers[0].update_config(cfg) if cfg.wandb else None
        print_meta_data(cfg, model, *loaders) if cfg.is_master else None

        wandb.watch(model, log='all', log_freq=100)

        # Train
        train_engine = create_train_engine(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler,
                                           (start_epoch, n_epochs))

        df = train_engine()

        df = pd.concat(
            [df, pd.DataFrame({'Data': 'ODDAccuracy', 'Acc': df[~df.Data.isin(['imagenet', 'Mean'])]['Acc'].mean()},
                              index=[0])])
        df.loc[df['Data'] == 'Mean', 'Data'] = 'EvalAccuracy'

        df = df.to_dict(orient='list')
        _log = dict()
        for k, v in zip(df['Data'], df['Acc']):
            _log[k] = v
        wandb.log(_log)

        # Eval
        # cfg.train.batch_size = 4096
        # model.eval()
        #
        # cfg.dataset.name = 'imagenet_ds'
        # acc_list = list()
        # for k, v in dataset2dict(cfg.dataset).items():
        #     cfg.dataset = v
        #     ds_backbone = cfg.model.backbone.split('-')[-1]
        #     train_dataset = create_dataset(cfg.dataset, is_train=True, split=cfg.dataset.train, backbone=ds_backbone)
        #     test_dataset = create_dataset(cfg.dataset, is_train=False, split=cfg.dataset.test, backbone=ds_backbone)
        #
        #     engine = create_task_engine(cfg, fabric, model, tokenizer, train_dataset, test_dataset)
        #     metrics = engine(n_shots=to_list(cfg.n_shot))
        #     acc = float(metrics['simple_adapter_classification'])
        #     acc_list.append(acc)
        #     wandb.log({f'{k}_acc': acc})
        #
        # wandb.log({'OODAccuracy': sum(acc_list[1:]) / len(acc_list[1:])})
        # wandb.log({'EvalAccuracy': sum(acc_list) / len(acc_list)})


if __name__ == "__main__":
    main()
