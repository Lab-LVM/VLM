# Epoch Ablation
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=5 name=B32_E5 gpu='[0,1,2,3]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=15 name=B32_E15 gpu='[4,5,6,7]'

# Augmentation ablation Dual
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetAA dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=5 name=B32_E5_Ablation_AA gpu='[0,1,2,3]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetAN dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=5 name=B32_E5_Ablation_AN gpu='[4,5,6,7]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetNN dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=5 name=B32_E5_Ablation_NN gpu='[4,5,6,7]'

# Augmentation ablation Single
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetsimText dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=5 name=B32_E5_Ablation_Sim gpu='[0,1,2,3]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetsimAug dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=5 name=B32_E5_Ablation_SimAug gpu='[0,1,2,3]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetsimAugNorm dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=5 name=B32_E5_Ablation_SimAugNorm gpu='[0,1,2,3]'
