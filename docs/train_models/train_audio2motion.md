# 0.Get pre-trained models & Data
- Get the Binarized dataset following `docs/process_data/process_th1kh.md`. You will see `data/binary/th1kh/train.data`

# 1. Train audio_lm3d_syncnet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tasks/run.py --config=egs/os_avatar/audio_lm3d_syncnet.yaml --exp_name=audio_lm3d_syncnet --reset


# 2. Train audio2motion model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tasks/run.py --config=egs/os_avatar/audio2motion_vae.yaml --exp_name=audio2motion_vae --hparams=syncnet_ckpt_dir=checkpoints/audio_lm3d_syncnet --reset

# 3.Inference
- See `README.md`, change the name of checkpoint to your own audio2motion_vae model.
