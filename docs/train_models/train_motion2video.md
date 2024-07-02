# 0.Get pre-trained models & Data
- Get the Binarized dataset following `docs/process_data/process_th1kh.md`. You will see `data/binary/th1kh/train.data`
- Download `pretrained_ckpts.zip` in this [Google Drive](https://drive.google.com/drive/folders/1MAveJf7RvJ-Opg1f5qhLdoRoC_Gc6nD9?usp=sharing), unzip it and place it into `checkpoints/pretrained_ckpts`. You will see `checkpoints/pretrained_ckpts/mit_b0.pth` and `checkpoints/pretrained_ckpts/eg3d_baseline_run2`.


# 1. Train Img-to-Plane Model
## 1.1 image-to-triplane model in real3d-portrait
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tasks/run.py --config=egs/os_avatar/img2plane.yaml --hparams=triplane_feature_type=triplane --exp_name=img2plane --reset
```
## 1.2 image-to-grid model in zera-portrait (Recommended)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tasks/run.py --config=egs/os_avatar/img2plane.yaml --exp_name=img2grid --reset
```

# 2.Train Motion-to-Video Model
```
# secc2plane_head
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tasks/run.py --config=egs/os_avatar/srcc_img2plane.yaml --exp_name=secc2plane --hparams=init_from_ckpt=checkpoints/img2grid --reset

# secc2plane_torso
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tasks/run.py --config=egs/os_avatar/srcc_img2plane_torso.yaml --exp_name=secc2plane_torso --hparams=init_from_ckpt=checkpoints/secc2plane --reset
```

# 3.Inference
- See `README.md`, change the name of checkpoint to your own secc2plane_torso model.
