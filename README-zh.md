# Real3D-Portrait: One-shot Realistic 3D Talking Portrait Synthesis
[![arXiv](https://img.shields.io/badge/arXiv-Paper-%3CCOLOR%3E.svg)](https://arxiv.org/abs/2401.08503)| [![GitHub Stars](https://img.shields.io/github/stars/yerfor/Real3DPortrait
)](https://github.com/yerfor/Real3DPortrait) | [![downloads](https://img.shields.io/github/downloads/yerfor/Real3DPortrait/total.svg)](https://github.com/yerfor/Real3DPortrait/releases) | ![visitors](https://visitor-badge.glitch.me/badge?page_id=yerfor/Real3DPortrait)

[English Readme](./README.md)

这个仓库是Real3D-Portrait的官方PyTorch实现，用于实现单参考图(one-shot)、高视频真实度(video reality)的虚拟人视频合成。您可以访问我们的[项目页面](https://real3dportrait.github.io/)以观看Demo视频, 阅读我们的[论文](https://arxiv.org/pdf/2401.08503.pdf)以了解技术细节。

<p align="center">
    <br>
    <img src="assets/real3dportrait.png" width="100%"/>
    <br>
</p>

# 快速上手！
## 安装环境
请参照[环境配置文档](docs/prepare_env/install_guide-zh.md)，配置Conda环境`real3dportrait`
## 下载预训练与第三方模型
### 3DMM BFM模型
下载3DMM BFM模型：[Google Drive](https://drive.google.com/drive/folders/1o4t5YIw7w4cMUN4bgU9nPf6IyWVG1bEk?usp=sharing) 或 [BaiduYun Disk](https://pan.baidu.com/s/1aqv1z_qZ23Vp2VP4uxxblQ?pwd=m9q5 ) 提取码: m9q5


下载完成后，放置全部的文件到`deep_3drecon/BFM`里，文件结构如下：
```
deep_3drecon/BFM/
├── 01_MorphableModel.mat
├── BFM_exp_idx.mat
├── BFM_front_idx.mat
├── BFM_model_front.mat
├── Exp_Pca.bin
├── facemodel_info.mat
├── index_mp468_from_mesh35709.npy
├── mediapipe_in_bfm53201.npy
└── std_exp.txt
```

### 预训练模型
下载预训练的Real3D-Portrait：[Google Drive](https://drive.google.com/drive/folders/1MAveJf7RvJ-Opg1f5qhLdoRoC_Gc6nD9?usp=sharing) 或 [BaiduYun Disk](https://pan.baidu.com/s/1Mjmbn0UtA1Zm9owZ7zWNgQ?pwd=6x4f ) 提取码: 6x4f
  
下载完成后，放置全部的文件到`checkpoints`里并解压，文件结构如下：
```
checkpoints/
├── 240126_real3dportrait_orig
│   ├── audio2secc_vae
│   │   ├── config.yaml
│   │   └── model_ckpt_steps_400000.ckpt
│   └── secc2plane_torso_orig
│       ├── config.yaml
│       └── model_ckpt_steps_100000.ckpt
└── pretrained_ckpts
    └── mit_b0.pth
```

## 推理测试
我们目前提供了**命令行（CLI）**与**Gradio WebUI**推理方式，并将在未来提供Google Colab方式。我们同时支持音频驱动（Audio-Driven）与视频驱动（Video-Driven）：

- 音频驱动场景下，需要至少提供`source image`与`driving audio`
- 视频驱动场景下，需要至少提供`source image`与`driving expression video`

### Gradio WebUI推理
启动Gradio WebUI，按照提示上传素材，点击`Generate`按钮即可推理：
```bash
python inference/app_real3dportrait.py
```

### 命令行推理
首先，切换至项目根目录并启用Conda环境：
```bash
cd <Real3DPortraitRoot>
conda activate real3dportrait
export PYTHON_PATH=./
```
音频驱动场景下，需要至少提供source image与driving audio，推理指令：
```bash
python inference/real3d_infer.py \
--src_img <PATH_TO_SOURCE_IMAGE> \
--drv_aud <PATH_TO_AUDIO> \
--drv_pose <PATH_TO_POSE_VIDEO, OPTIONAL> \
--bg_img <PATH_TO_BACKGROUND_IMAGE, OPTIONAL> \
--out_name <PATH_TO_OUTPUT_VIDEO, OPTIONAL>
```
视频驱动场景下，需要至少提供source image与driving expression video（作为drv_aud参数），推理指令：
```bash
python inference/real3d_infer.py \
--src_img <PATH_TO_SOURCE_IMAGE> \
--drv_aud <PATH_TO_EXP_VIDEO> \
--drv_pose <PATH_TO_POSE_VIDEO, OPTIONAL> \
--bg_img <PATH_TO_BACKGROUND_IMAGE, OPTIONAL> \
--out_name <PATH_TO_OUTPUT_VIDEO, OPTIONAL>
```
一些可选参数注释：
- `--drv_pose` 指定时提供了运动pose信息，不指定则为静态运动
- `--bg_img` 指定时提供了背景信息，不指定则为source image提取的背景
- `--mouth_amp` 嘴部张幅参数，值越大张幅越大
- `--map_to_init_pose` 值为`True`时，首帧的pose将被映射到source pose，后续帧也作相同变换
- `--temperature` 代表audio2motion的采样温度，值越大结果越多样，但同时精确度越低
- `--out_name` 不指定时，结果将保存在`infer_out/tmp/`中
- `--out_mode` 值为`final`时，只输出说话人视频；值为`concat_debug`时，同时输出一些可视化的中间结果

指令示例：
```bash
python inference/real3d_infer.py \
--src_img data/raw/examples/Macron.png \
--drv_aud data/raw/examples/Obama_5s.wav \
--drv_pose data/raw/examples/May_5s.mp4 \
--bg_img data/raw/examples/bg.png \
--out_name output.mp4 \
--out_mode concat_debug
```

## ToDo
- [x] **Release Pre-trained weights of Real3D-Portrait.**
- [x] **Release Inference Code of Real3D-Portrait.**
- [x] **Release Gradio Demo of Real3D-Portrait..**
- [ ] **Release Google Colab of Real3D-Portrait..**
- [ ] **Release Training Code of Real3D-Portrait.**

# 引用我们
如果这个仓库对你有帮助，请考虑引用我们的工作：
```
@article{ye2024real3d,
  title={Real3D-Portrait: One-shot Realistic 3D Talking Portrait Synthesis},
  author={Ye, Zhenhui and Zhong, Tianyun and Ren, Yi and Yang, Jiaqi and Li, Weichuang and Huang, Jiawei and Jiang, Ziyue and He, Jinzheng and Huang, Rongjie and Liu, Jinglin and others},
  journal={arXiv preprint arXiv:2401.08503},
  year={2024}
}
@article{ye2023geneface++,
  title={GeneFace++: Generalized and Stable Real-Time Audio-Driven 3D Talking Face Generation},
  author={Ye, Zhenhui and He, Jinzheng and Jiang, Ziyue and Huang, Rongjie and Huang, Jiawei and Liu, Jinglin and Ren, Yi and Yin, Xiang and Ma, Zejun and Zhao, Zhou},
  journal={arXiv preprint arXiv:2305.00787},
  year={2023}
}
@article{ye2023geneface,
  title={GeneFace: Generalized and High-Fidelity Audio-Driven 3D Talking Face Synthesis},
  author={Ye, Zhenhui and Jiang, Ziyue and Ren, Yi and Liu, Jinglin and He, Jinzheng and Zhao, Zhou},
  journal={arXiv preprint arXiv:2301.13430},
  year={2023}
}
```