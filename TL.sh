export _DS_NAME=flowers102
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0

export _DS_NAME=pcam
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0

export _DS_NAME=caltech101
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0

export _DS_NAME=stanfordcars
python main_full.py +setup=our2 train.epochs=100 dataset=${_DS_NAME} dataset.name=${_DS_NAME}ra dataset@eval_dataset=${_DS_NAME} train.optimizer.weight_decay=0.0

python main_full.py +setup=our2_iwild dataset=iwildcam dataset.name=iwildcamra dataset@eval_dataset=iwildcam train.epochs=20 name=iWilCam_E20
#python main_full.py +setup=our2_iwild dataset=iwildcam dataset.name=iwildcamra dataset@eval_dataset=iwildcam train.epochs=50 name=iWilCam_E50
#python main_full.py +setup=our2_iwild dataset=iwildcam dataset.name=iwildcamra dataset@eval_dataset=iwildcam train.epochs=40 name=iWilCam_E40