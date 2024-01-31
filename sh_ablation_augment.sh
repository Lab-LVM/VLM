# Default B-16
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds train.epochs=5 name=B16_E5
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds train.epochs=10 name=B16_E10
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds train.epochs=15 name=B16_E15
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds train.epochs=20 name=B16_E20

# Default B-32
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=10 name=B32_E10 gpu='[0,1,2,3]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=15 name=B32_E15 gpu='[0,1,2,3]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=20 name=B32_E20 gpu='[0,1,2,3]'

# Epoch Ablation
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=5 name=B32_E5 gpu='[0,1,2,3]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetraTextOri dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=15 name=B32_E15 gpu='[4,5,6,7]'

# Augmentation ablation Dual
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetAA dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=15 name=B32_E15_Ablation_AA gpu='[0,1,2,3]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetAN dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=15 name=B32_E15_Ablation_AN_SoftCE gpu='[0,1,2,3]' train.criterion=SoftCE
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetNN dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=15 name=B32_E15_Ablation_NN_SoftCE gpu='[0,1,2,3]' train.criterion=SoftCE

# Augmentation ablation Single
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetsimText dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=15 name=B32_E15_Ablation_Sim gpu='[0,1,2,3]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetsimAug dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=15 name=B32_E15_Ablation_SimAug gpu='[0,1,2,3]'
python main_full_imagenet.py +setup=our2 train=base_train_pre dataset.name=imagenetsimAugNorm dataset@eval_dataset=imagenet_ds model.backbone=ViT-B32 train.batch_size=128 train.num_workers=12 train.adapter_lr=0.001 train.epochs=15 name=B32_E15_Ablation_SimAugNorm gpu='[0,1,2,3]'

