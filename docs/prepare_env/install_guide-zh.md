# 环境配置
[English Doc](./install_guide.md)

本文档陈述了搭建Real3D-Portrait Python环境的步骤，我们使用了Conda来管理依赖。

以下配置已在 A100/V100 + CUDA11.7 中进行了验证。


# 1. 安装CUDA
我们推荐安装CUDA `11.7`，其他CUDA版本（例如`10.2`、`12.x`）也可能有效。 

# 2. 安装Python依赖
```
cd <Real3DPortraitRoot>
source <CondaRoot>/bin/activate
conda create -n real3dportrait python=3.9
conda activate real3dportrait

# 我们推荐安装torch2.0.1+cuda11.7. 
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# 从源代码安装，需要比较长的时间 (如果遇到各种time-out问题，建议使用代理)
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 利用conda安装pytorch (For fast installation, Linux only)
conda install pytorch3d::pytorch3d
## 如果conda安装失败,一个兼容性的选择是从Github拉取源码并本地编译
## 这可能会花费较长时间（可能数十分钟左右）；由于要连接Github，可能经常面临time-out问题，请考虑使用代理。
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# MMCV安装
pip install cython
pip install openmim==0.3.9
mim install mmcv==2.1.0 # 使用mim来加速mmcv安装

# 其他依赖项
pip install -r docs/prepare_env/requirements.txt -v


如果你遇到如下错误，请尝试使用以下命令安装依赖项：
pip install -r docs/prepare_env/requirements.txt -v --use-deprecated=legacy-resolver

> ERROR: pip's dependency resolver does not currently take into account all the packages
> that are installed. This behaviour is the source of the following dependency conflicts.
> openxlab 0.0.34 requires setuptools~=60.2.0, but you have setuptools 69.1.1 which is incompatible.


```

