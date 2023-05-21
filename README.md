Utilities for converting pretrained checkpoints from other codebases for `jaxrl_m.vision`

#### Resnetv2-50

Convert checkpoints from https://github.com/google-research/big_transfer using commands like:

```
python pretrained_vision/pretrained_vision/bigvision_resnetv2.py --pretrained_path=BiT-M-R50x1.npz --prefix=BiT-M-R50x1
```

#### Resnetv1-50

Convert pytorch resnet-50 checkpoints using commands like:

```
# If a .pt file containing the state dict:
python pretrained_vision/pretrained_vision/resnetv1.py --pretrained_path=my_encoder.pt --prefix=my_encoder

# Or specify among `r3m`, `imagenet` or `vip`
python pretrained_vision/pretrained_vision/resnetv1.py --pretrained_path=imagenet --prefix=imagenet-resnetv1-50
```

#### ViT

Convert checkpoints from https://github.com/google-research/vision_transformer/

```
python pretrained_vision/pretrained_vision/vit.py --pretrained_path=S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --prefix=vit_s16_augreg --encoder='ViT-S/16'
```


#### MAE

Convert checkpoints compatible with https://github.com/facebookresearch/mae 

```
python pretrained_vision/pretrained_vision/mae.py --pretrained_path=notebooks/mae_visualize_vit_base.pth --prefix=mae_base --encoder='mae_base'
```