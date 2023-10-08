import os

import torch
from lightning import Fabric
from termcolor import colored

from .callback import CallBack
from .logger import WandBNCSVLogger


def seed_everything(seed, local_rank=0, workers=0):
    import random
    import numpy

    each_seed = seed + local_rank
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    random.seed(each_seed)
    numpy.random.seed(each_seed)
    torch.manual_seed(each_seed)
    torch.cuda.manual_seed_all(each_seed)


def setup_cuda(gpu):
    if isinstance(gpu, int):
        gpu = [gpu]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True # If deterministic = True, training time will be increased.
    return gpu


def setup_fabric(cfg):
    gpu = setup_cuda(cfg.gpu)

    if len(gpu) == 1 and cfg.strategy != 'auto':
        print(colored('[Config Change]', 'yellow'), f'{cfg.strategy} is turned off. Running on single strategy')
        cfg.strategy = 'auto'

    fabric = Fabric(accelerator=cfg.accelerator,
                    strategy=cfg.strategy,
                    devices=-1 if gpu is not None else None,
                    precision=cfg.precision,
                    num_nodes=1,
                    loggers=[WandBNCSVLogger(cfg)],
                    callbacks=[CallBack]
                    )
    fabric.launch()
    seed_everything(cfg.train.seed)

    cfg.is_master = True if fabric.global_rank == 0 else False
    cfg.world_size = fabric.world_size
    cfg.distributed = torch.distributed.is_initialized()

    return fabric
