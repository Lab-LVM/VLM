from typing import Dict, Any, Callable

_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}

_FEATURE_REGISTRY: Dict[str, Callable[..., Any]] = {}
_TASK_REGISTRY: Dict[str, Callable[..., Any]] = {}
_TRAIN_REGISTRY: Dict[str, Callable[..., Any]] = {}


def list_registry():
    res = f'Model: {_MODEL_REGISTRY.keys()}\n'
    res += f'Task Engine: {_TASK_REGISTRY.keys()}\n'
    res += f'Feature Engine: {_FEATURE_REGISTRY.keys()}\n'
    res += f'Train Engine: {_TRAIN_REGISTRY.keys()}\n'
    return res


def register_model(fn):
    _MODEL_REGISTRY[fn.__name__] = fn
    return fn


def register_task_engine(fn):
    _TASK_REGISTRY[fn.__name__] = fn
    return fn


def register_train_engine(fn):
    _TRAIN_REGISTRY[fn.__name__] = fn
    return fn


def register_feature_engine(fn):
    _FEATURE_REGISTRY[fn.__name__] = fn
    return fn


def create_model(model_name, **kwargs):
    fn = _MODEL_REGISTRY.get(model_name, None)
    if fn:
        return fn(**kwargs)
    else:
        raise NotImplementedError(f'{model_name} model is not implemented.')


def create_task_engine(cfg, fabric, model, tokenizer, train_dataset, val_dataset, **kwargs):
    engine_name = cfg.model.task_engine
    fn = _TASK_REGISTRY.get(engine_name, None)
    if fn:
        return fn(cfg, fabric, model, tokenizer, train_dataset, val_dataset, **kwargs)
    else:
        raise NotImplementedError(f'{engine_name} engine is not implemented.')


def create_train_engine(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs, **kwargs):
    engine_name = cfg.model.train_engine
    fn = _TRAIN_REGISTRY.get(engine_name, None)
    if fn:
        return fn(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs, **kwargs)
    else:
        raise NotImplementedError(f'{engine_name} train engine is not implemented.')
