export _DS_NAME=flowers102
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0

export _DS_NAME=pcam
python main_full.py +setup=our2 train.num_workers=4 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0

export _DS_NAME=caltech101
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0

export _DS_NAME=stanfordcars
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0

python main_full.py +setup=our2_iwild dataset=iwildcam dataset.name=iwildcamra dataset@eval_dataset=iwildcam model.backbone=ViT-B32 train.epochs=20 name=B32_iWilCam_E20 gpu='[0,1,2,3]' train.batch_size=128 train.adapter_lr=1e-2
python main_full.py +setup=our2_iwild dataset=iwildcam dataset.name=iwildcamra dataset@eval_dataset=iwildcam model.backbone=ViT-B32 train.epochs=50 name=B32_iWilCam_E50 gpu='[4,5,6,7]' train.batch_size=128 train.adapter_lr=1e-2
#python main_full.py +setup=our2_iwild dataset=iwildcam dataset.name=iwildcamra dataset@eval_dataset=iwildcam train.epochs=40 name=iWilCam_E40