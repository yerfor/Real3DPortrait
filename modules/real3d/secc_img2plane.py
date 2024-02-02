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
import math
from modules.real3d.segformer import SegFormerImg2PlaneBackbone, SegFormerSECC2PlaneBackbone
from modules.real3d.img2plane_baseline import OSAvatar_Img2plane
from modules.img2plane.img2plane_model import Img2PlaneModel
from utils.commons.hparams import hparams
# 换成attention吧？value用plane。

class OSAvatarSECC_Img2plane(OSAvatar_Img2plane):
    def __init__(self,  hp=None):
        super().__init__(hp=hp)
        hparams = self.hparams
        # extract canonical triplane from src img
        self.cano_img2plane_backbone = self.img2plane_backbone # rename
        del self.img2plane_backbone
        self.secc_img2plane_backbone = SegFormerSECC2PlaneBackbone(mode=hparams['secc_segformer_scale'], out_channels=3*self.triplane_hid_dim*self.triplane_depth, pncc_cond_mode=hparams['pncc_cond_mode'])
        self.lambda_pertube_blink_secc = torch.nn.Parameter(torch.tensor([0.001]), requires_grad=False)
        self.lambda_pertube_secc = torch.nn.Parameter(torch.tensor([0.001]), requires_grad=False)

    def on_train_full_model(self):
        self.requires_grad_(True)
            
    def on_train_nerf(self):
        self.cano_img2plane_backbone.requires_grad_(True)
        self.secc_img2plane_backbone.requires_grad_(True)
        self.decoder.requires_grad_(True)
        self.superresolution.requires_grad_(False)

    def on_train_superresolution(self):
        self.cano_img2plane_backbone.requires_grad_(False)
        self.secc_img2plane_backbone.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.superresolution.requires_grad_(True)

    def cal_cano_plane(self, img, cond=None, **kwargs):
        hparams = self.hparams
        planes = cano_planes = self.cano_img2plane_backbone(img, cond, **kwargs)  # [B, 3, C*D, H, W]
        if hparams.get("triplane_feature_type", "triplane") in ['triplane', 'trigrid']:
            planes = planes.view(len(planes), 3, self.triplane_hid_dim*self.triplane_depth, planes.shape[-2], planes.shape[-1])
        elif hparams.get("triplane_feature_type", "triplane") in ['trigrid_v2']:
            b, k, cd, h, w = planes.shape # k = 3
            planes = planes.reshape([b, k*cd, h, w])
            planes = self.plane2grid_module(planes)
            planes = planes.reshape([b, k, cd, h, w])
        else:
            raise NotImplementedError()
        return planes

    def cal_secc_plane(self, cond):
        cano_pncc, src_pncc, tgt_pncc = cond['cond_cano'], cond['cond_src'], cond['cond_tgt']
        if self.hparams.get("pncc_cond_mode", "cano_tgt") == 'cano_src_tgt':
            inp_pncc = torch.cat([cano_pncc, src_pncc, tgt_pncc], dim=1)
        else:
            inp_pncc = torch.cat([cano_pncc, tgt_pncc], dim=1)
        secc_planes = self.secc_img2plane_backbone(inp_pncc)
        return secc_planes
    
    def cal_plane_given_cano(self, cano_planes, cond=None):
        # cano_planes: # [B, 3, C*D, H, W]
        secc_planes = self.cal_secc_plane(cond) # [B, 3, C*D, H, W]
        if self.hparams.get("phase1_plane_fusion_mode", "add") == 'add':
            planes = cano_planes + secc_planes
        elif self.hparams.get("phase1_plane_fusion_mode", "add") == 'mul':
            planes = cano_planes * secc_planes
        else: raise NotImplementedError()
        return planes

    def cal_plane(self, img, cond, ret=None, **kwargs):
        cano_planes = self.cal_cano_plane(img, cond, **kwargs) # [B, 3, C*D, H, W]
        planes = self.cal_plane_given_cano(cano_planes, cond)
        return planes, cano_planes

    def sample(self, coordinates, directions, img, cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, ref_camera=None, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        planes, _ = self.cal_plane(img, cond, ret={}, ref_camera=ref_camera)
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def synthesis(self, img, camera, cond=None, ret=None, update_emas=False, cache_backbone=True, use_cached_backbone=False, **synthesis_kwargs):
        if ret is None: ret = {}
        cam2world_matrix = camera[:, :16].view(-1, 4, 4)
        intrinsics = camera[:, 16:25].view(-1, 3, 3)

        neural_rendering_resolution = self.neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone:
            # use the cached cano_planes obtained from a previous forward with flag cache_backbone=True
            cano_planes = self._last_cano_planes
            planes = self.cal_plane_given_cano(cano_planes, cond)
        else:
            planes, cano_planes = self.cal_plane(img, cond, ret, **synthesis_kwargs)
        if cache_backbone:
            self._last_cano_planes = cano_planes

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, is_ray_valid = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        weights_image = weights_samples.permute(0, 2, 1).reshape(N,1,H,W).contiguous() # [N,1,H,W]
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        if self.hparams.get("mask_invalid_rays", False):
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
