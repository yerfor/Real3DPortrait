# Prepare the Environment
[中文文档](./install_guide-zh.md)

This guide is about building a python environment for Real3D-Portrait with Conda.

The following installation process is verified in A100/V100 + CUDA11.7.


# 1. Install CUDA
We use CUDA extensions from [torch-ngp](https://github.com/ashawkey/torch-ngp), you may need to manually install CUDA from the [offcial page](https://developer.nvidia.com/cuda-toolkit). We recommend to install CUDA `11.7` (which is verified in various types of GPUs), but other CUDA versions (such as `10.2`) may also work well. Make sure your cuda path (typically `/usr/local/cuda`) points to a installed `/usr/local/cuda-11.7`. Note that we cannot support CUDA 12.* currently. 

# 2. Install Python Packages
```
cd <Real3DPortraitRoot>
source <CondaRoot>/bin/activate
conda create -n real3dportrait python=3.9
conda activate real3dportrait
conda install conda-forge::ffmpeg # ffmpeg with libx264 codec to turn images to video

### We recommend torch2.0.1+cuda11.7. We found torch=2.1+cuda12.1 leads to erors in torch-ngp
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Build from source, it may take a long time (Proxy is recommended if encountering the time-out problem)
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# MMCV for some network structure
pip install cython
pip install openmim==0.3.9
mim install mmcv==2.1.0 # use mim to speed up installation for mmcv

# other dependencies
pip install -r docs/prepare_env/requirements.txt -v

```
