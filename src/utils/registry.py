from typing import Dict, Any, Callable

_ENGINE_REGISTRY: Dict[str, Callable[..., Any]] = {}
_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}
_TRAIN_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_engine(fn):
    _ENGINE_REGISTRY[fn.__name__] = fn
    return fn


def register_train(fn):
    _TRAIN_REGISTRY[fn.__name__] = fn
    return fn


def register_model(fn):
    _MODEL_REGISTRY[fn.__name__] = fn
    return fn


def create_model(model_name, pretrained, finetuned, **kwargs):
    fn = _MODEL_REGISTRY.get(model_name, None)
    if fn:
        return fn(pretrained=pretrained, finetuned=finetuned, **kwargs)
    else:
        raise NotImplementedError(f'{model_name} model is not implemented.')


def create_engine(cfg, fabric, model, tokenizer, train_dataset, val_dataset, **kwargs):
    engine_name = cfg.model.model_name + 'Engine'
    fn = _ENGINE_REGISTRY.get(engine_name, None)
    if fn:
        return fn(cfg, fabric, model, tokenizer, train_dataset, val_dataset, **kwargs)
    else:
        raise NotImplementedError(f'{engine_name} engine is not implemented.')


def create_train_engine(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs, **kwargs):
    train_engine_name = cfg.model.model_name + 'FineTuneEngine'
    fn = _TRAIN_REGISTRY.get(train_engine_name, None)
    if fn:
        return fn(cfg, fabric, model, tokenizer, loaders, criterion, optimizer, scheduler, epochs, **kwargs)
    else:
        raise NotImplementedError(f'{train_engine_name} train engine is not implemented.')
