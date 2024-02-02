# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np
import torch
import copy
from modules.real3d.segformer import SegFormerImg2PlaneBackbone
from modules.img2plane.triplane import OSGDecoder
from modules.eg3ds.models.superresolution import SuperresolutionHybrid8XDC
from modules.eg3ds.volumetric_rendering.renderer import ImportanceRenderer
from modules.eg3ds.volumetric_rendering.ray_sampler import RaySampler
from modules.img2plane.img2plane_model import Img2PlaneModel

from utils.commons.hparams import hparams
import torch.nn.functional as F
import torch.nn as nn
from modules.real3d.facev2v_warp.layers import *
from einops import rearrange


class SameBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size=3, padding=1):
        super(SameBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding, padding_mode='replicate')
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding, padding_mode='replicate')
        self.norm1 = nn.GroupNorm(4, in_features, affine=True)
        self.norm2 = nn.GroupNorm(4, in_features, affine=True)
        self.alpha = nn.Parameter(torch.tensor([0.01]))

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = x + self.alpha * out
        return out


class Plane2GridModule(nn.Module):
    def __init__(self, triplane_depth=3, in_out_dim=96):
        super().__init__()
        self.triplane_depth = triplane_depth
        self.in_out_dim = in_out_dim
        if self.triplane_depth <= 3:
            self.num_layers_per_block = 1
        else:
            self.num_layers_per_block = 2
        self.res_blocks_3d = nn.Sequential(*[SameBlock3d(in_out_dim//3) for _ in range(self.num_layers_per_block)])
        
    def forward(self, x):
        x_inp = x # [1, 96*D, H, W]
        N, KCD, H, W = x.shape
        K, C, D = 3, KCD // self.triplane_depth // 3, self.triplane_depth
        assert C == self.in_out_dim // 3
        x = rearrange(x, 'n (k c d) h w -> (n k) c d h w', k=K, c=C, d=D) # ==> [1, 96, D, H, W]
        x = self.res_blocks_3d(x) # ==> [1, 96, D, H, W]
        x = rearrange(x, '(n k) c d h w -> n (k c d) h w', k=K)
        return x


class OSAvatar_Img2plane(torch.nn.Module):
    def __init__(self, hp=None):
        super().__init__()
        global hparams
        self.hparams = copy.copy(hparams) if hp is None else copy.copy(hp)
        hparams = self.hparams

        self.camera_dim = 25 # extrinsic 4x4 + intrinsic 3x3
        self.neural_rendering_resolution = hparams.get("neural_rendering_resolution", 128)
        self.w_dim = hparams['w_dim']
        self.img_resolution = hparams['final_resolution']
        self.triplane_depth = hparams.get("triplane_depth", 1)
        
        self.triplane_hid_dim = triplane_hid_dim = hparams.get("triplane_hid_dim", 32)
        # extract canonical triplane from src img
        self.img2plane_backbone = Img2PlaneModel(out_channels=3*triplane_hid_dim*self.triplane_depth, hp=hparams)
        if hparams.get("triplane_feature_type", "triplane") in ['trigrid_v2']:
            self.plane2grid_module = Plane2GridModule(triplane_depth=self.triplane_depth, in_out_dim=3*triplane_hid_dim) # add depth here
          
        # positional embedding
        self.decoder = OSGDecoder(triplane_hid_dim, {'decoder_lr_mul': 1, 'decoder_output_dim': triplane_hid_dim})
        # create super resolution network
        self.sr_num_fp16_res = 0
        self.sr_kwargs = {'channel_base': hparams['base_channel'], 'channel_max': hparams['max_channel'], 'fused_modconv_default': 'inference_only'}
        self.superresolution = SuperresolutionHybrid8XDC(channels=triplane_hid_dim, img_resolution=self.img_resolution, sr_num_fp16_res=self.sr_num_fp16_res, sr_antialias=True, large_sr=hparams.get('large_sr',False), **self.sr_kwargs)
        # Rendering Options
        self.renderer = ImportanceRenderer(hp=hparams)
        self.ray_sampler = RaySampler()
        self.rendering_kwargs = {'image_resolution': hparams['final_resolution'], 
                            'disparity_space_sampling': False, 
                            'clamp_mode': 'softplus',
                            'gpc_reg_prob': hparams['gpc_reg_prob'], 
                            'c_scale': 1.0, 
                            'superresolution_noise_mode': 'none', 
                            'density_reg': hparams['lambda_density_reg'], 'density_reg_p_dist': hparams['density_reg_p_dist'], 
                            'reg_type': 'l1', 'decoder_lr_mul': 1.0, 
                            'sr_antialias': True, 
                            'depth_resolution': hparams['num_samples_coarse'], 
                            'depth_resolution_importance': hparams['num_samples_fine'],
                            'ray_start': 'auto', 'ray_end': 'auto',
                            'box_warp': hparams.get("box_warp", 1.), # 3DMM坐标系==world坐标系，而3DMM的landmark的坐标均位于[-1,1]内
                            'avg_camera_radius': 2.7,
                            'avg_camera_pivot': [0, 0, 0.2],
                            'white_back': False,
                            }

    def cal_plane(self, img, cond=None, ret=None, **synthesis_kwargs):
        hparams = self.hparams
        planes = self.img2plane_backbone(img, cond, **synthesis_kwargs) #  [B, 3, C*D, H, W]
        if hparams.get("triplane_feature_type", "triplane") in ['triplane', 'trigrid']:
            planes = planes.view(len(planes), 3, self.triplane_hid_dim*self.triplane_depth, planes.shape[-2], planes.shape[-1])
        elif hparams.get("triplane_feature_type", "triplane") in ['trigrid_v2']:
            b, k, cd, h, w = planes.shape
            planes = planes.reshape([b, k*cd, h, w])
            planes = self.plane2grid_module(planes)
            planes = planes.reshape([b, k, cd, h, w])
        else:
            raise NotImplementedError()
        return planes # [B, 3, C*D, H, W]
    
    def _forward_sr(self, rgb_image, feature_image, cond, ret, **synthesis_kwargs):
        hparams = self.hparams
        ones_ws = torch.ones([feature_image.shape[0], 14, hparams['w_dim']], dtype=feature_image.dtype, device=feature_image.device)
        if hparams.get("sr_type", "vanilla") == 'vanilla':
            sr_image = self.superresolution(rgb_image, feature_image, ones_ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        elif hparams.get("sr_type", "vanilla") == 'spade':
            sr_image = self.superresolution(rgb_image, feature_image, ones_ws, segmap=cond['ref_head_img'], noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        return sr_image

    def synthesis(self, img, camera, cond=None, ret=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        hparams = self.hparams
        if ret is None: ret = {}
        cam2world_matrix = camera[:, :16].view(-1, 4, 4)
        intrinsics = camera[:, 16:25].view(-1, 3, 3)

        neural_rendering_resolution = self.neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.cal_plane(img, cond, ret, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes
        
        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, is_ray_valid = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        weights_image = weights_samples.permute(0, 2, 1).reshape(N,1,H,W).contiguous() # [N,1,H,W]
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        if hparams.get("mask_invalid_rays", False):
            is_ray_valid_mask = is_ray_valid.reshape([feature_samples.shape[0], 1,self.neural_rendering_resolution,self.neural_rendering_resolution]) # [B, 1, H, W]
            feature_image[~is_ray_valid_mask.repeat([1,feature_image.shape[1],1,1])] = -1
            # feature_image[~is_ray_valid_mask.repeat([1,feature_image.shape[1],1,1])] *= 0
            # feature_image[~is_ray_valid_mask.repeat([1,feature_image.shape[1],1,1])] -= 1
            depth_image[~is_ray_valid_mask] = depth_image[is_ray_valid_mask].min().item()

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        ret['weights_img'] = weights_image
        sr_image = self._forward_sr(rgb_image, feature_image, cond, ret, **synthesis_kwargs)
        rgb_image = rgb_image.clamp(-1,1)
        sr_image = sr_image.clamp(-1,1)
        ret.update({'image_raw': rgb_image, 'image_depth': depth_image, 'image': sr_image, 'image_feature': feature_image[:, 3:], 'plane': planes})
        return ret

    def sample(self, coordinates, directions, img, cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, ref_camera=None, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        planes = self.cal_plane(img, cond, ret={}, ref_camera=ref_camera)
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, img, camera, cond=None, ret=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, return_all=True, **synthesis_kwargs):
        # Render a batch of generated images.
        out = self.synthesis(img, camera, cond=cond, ret=ret, update_emas=update_emas, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        return out
