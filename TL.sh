export _DS_NAME=flowers102
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0 model.finetune=False name=Flower102_noAdapter
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} name=Flower102_noWD

export _DS_NAME=cifar100
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0 model.finetune=False name=CIFAR100_noAdapter
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} name=CIFAR100_noWD

export _DS_NAME=pcam
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0 model.finetune=False name=PCam_noAdapter
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} name=PCam_noWD
