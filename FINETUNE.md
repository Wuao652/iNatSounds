# iNat-SSL
The goal of this project is to pretrain an MAE model in the natural audio domain, eg. iNatSounds. The first step is to evaluate the existing work in audio-related MAE pretraining. We proposed to do both finetune and linearprob experiments on the iNatSounds dataset (use iNatSounds train split to finetune / linearprob the pretrained model, and use iNatSounds test split to evaluate the final model).
### Fine-tuning ImageNet pretrained ViT
This is already done in the NeurIPS paper, we simply change the packages used to define a vit-b model from torchvision to timm.

```
python3 main.py --spectrogram_dir /scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_new_spec \
                --json_dir /scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_release \
                --log_dir /scratch3/workspace/wuaoliu_umass_edu-inat_sounds/outputs/sound_id/mae_init \
                --seed 2 \
                --epochs 50 \
                --model vit_base_patch16 \
                --optim adam \
                --lr 1e-4 \
                --batch_size 256 \
                --exp_name imagenet_pretrained \
                --encoder_weight "" \
                --mode train \
                --mixup \
                --pretrained \
                --sound_aug \
```

### Fine-tuning ImageNet pretrained MAE
- Download the pretrained weights to the "mae_pretrain_weights" folder, see the facebook/mae github repo.
```
cd ./mae_pretrain_weights
wget ...
```
- Finetune on the iNatSounds dataset.
```
python3 main.py --spectrogram_dir /scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_new_spec \
                --json_dir /scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_release \
                --log_dir /scratch3/workspace/wuaoliu_umass_edu-inat_sounds/outputs/sound_id/mae_init \
                --seed 2 \
                --epochs 50 \
                --model vit_base_patch16 \
                --optim adam \
                --lr 1e-4 \
                --batch_size 256 \
                --exp_name mae_finetune \
                --encoder_weight /work/pi_gvanhorn_umass_edu/wuao/iNatSounds/mae_pretrain_weights/mae_pretrain_vit_base.pth \
                --mode train \
                --mixup \
                --pretrained \
                --sound_aug \
```
### Fine-tuning AudioSet pretrained AudioMAE
To be finished.


