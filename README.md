# Research Framework

My custom research framework based on **PyTorch + Fabric + Hydra + timm**.

## How to run

```bash
python main.py +setup= <SETUPNAME> model= <MODEL_NAME> gpu=[ 0 | 0,1 | 0,1,2,3,4,5 ]

# Examples
python main.py +setup=resnet50_a2 gpu=0,1 # Using resnet50_a2 setup
python main.py +setup=caformer model=caformer_s18 gpu=2,3,4,5 # Change the pre-defined model in model configs
python main.py +setup=deit model=deit_tiny_patch16_224 gpu=6,7 train.epochs=100  # Change the other configs(e.g., epochs, optimizer, etc.)
python main.py +setup=resnet50_cifar model.model_name=resnet34 gpu=8 # Change the model of timm
python main.py model.model_name=resnext50 dataset=cifar100 train.epochs=150 # Just using default setup
```

- [setup](configs/setup): pre-defined setup.
- [model](configs/model): pre-defined model setup.
    - OR, you can just use `model.model_name` of timm model(e.g., resnet50, deit_tiny_patch16_224, etc.)
- gpu: device id of gpu.

## Personal configs

- [info](configs/info/info.yaml): your wandb project information.
- [dataset.root](configs/dataset): you have to modify the root dir of each dataset.yaml
- [hydra.run.dir](configs/config.yaml): your default dir of each run is generated like the dir of `hydra.run`

## Performance

ImageNet

| Model                  | Paper | Ours   | Link |
|------------------------|-------|--------|------|
| resnet50_a3            |       | 0.7819 |      |
| resnet50_a2            |       | 0.7948 |      |
| deit_tiny_patch16_224  | 0.722 | 0.7255 |      |
| deit_small_patch16_224 | 0.798 | 0.7985 |      |
| deit_base_patch16_224  | 0.818 | -      |      |
| convnext_tiny          |       | 0.8266 |      |
| caformer_s18           |       | 0.8356 |      |

CIFAR100

| Model    | Paper | Ours   | Link |
|----------|-------|--------|------|
| resnet50 | -     | 0.8216 |      |
