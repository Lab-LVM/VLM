# VLM

## How to run

evaluation

```bash
python run_eval.py gpu=[ 0 | [0,1,2,3] ] model=[clip | tip] data=[ all | imagenet_ds | vlzb | dataset_name ] n_shot=[ 1 | [1,2] ]
```

train

```bash
python run_train.py +setup=[ clip | tip ] gpu=[ 0 | [0,1,2,3] ] n_shot=[ 0 | 1 | 2 ] # only single n_shot supported
```

## What you need to do to add a new model

To add your own model, you have to define model and engine.

### 1. Define Model

- Define your model in `src/models/YOUR_MODEL`.
- Register your model using `@register_model` and add it to `src/models/__init__.py`.

### 2. Define Engine

- Make `YOUR_MODEL.py` in `src/engine/model_engine`.
- Code `Feature Engine`, `Task Engine`, and `Train Engine`.
    - Feature Engine: Feature extractor from visual and language encoder of model
    - Task Engine: Task evaluator of downstream. Available task will be placed.
    - Train Engine: Engine for fine-tuning. If you don't have fine-tuned strategy, you can leave this.

### 3. Define configs

- Make model configs in `configs/model`
- If you have fine-tuned strategy, define the training configs in `configs/setup`.

## This repository is...

This repository is focused on fine-tuning VLM models.
The distinction between **Transfer Learning (TL)** and **Few Shot Learning (FSL)** is determined by `n_shot`.

When running `train.py`, where `n_shot=0` is considered TL and `n_shot>0` is considered FSL.