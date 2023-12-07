import os

import hydra


def resume(model, optimizer, scheduler, cfg, fabric):
    checkpoint_path = cfg.train.resume
    start_epoch = 0
    if checkpoint_path:
        start_epoch = load_checkpoint(model, checkpoint_path, optimizer, fabric)
        scheduler.step(start_epoch)
    return model, optimizer, scheduler, start_epoch


def load_checkpoint(model, checkpoint_path, optimizer=None, fabric=None):
    resume_epoch = None
    base_path = hydra.utils.get_original_cwd()

    if checkpoint_path.endswith('.ckpt'):
        checkpoint_path = os.path.join(base_path, checkpoint_path)
    else:
        checkpoint_path = os.path.join(base_path, checkpoint_path, 'latest.ckpt')

    if os.path.isfile(checkpoint_path):
        checkpoint = fabric.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            fabric.print('Restoring model state from checkpoint...')
            model.load_state_dict(checkpoint['state_dict'])

            if optimizer is not None and 'optimizer' in checkpoint:
                fabric.print('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch'] + 1

            fabric.print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            fabric.print("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        fabric.print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
