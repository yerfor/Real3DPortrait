# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import copy
import torch
import torch.nn as nn
from modules.eg3ds.models.networks_stylegan2 import FullyConnectedLayer
from modules.eg3ds.volumetric_rendering.renderer import ImportanceRenderer
from modules.eg3ds.volumetric_rendering.ray_sampler import RaySampler
from modules.eg3ds.models.superresolution import SuperresolutionHybrid2X, SuperresolutionHybrid4X, SuperresolutionHybrid8X, SuperresolutionHybrid8XDC

from modules.img2plane.img2plane_model import Img2PlaneModel
from utils.commons.hparams import hparams


class Img2TriPlaneGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__(hp=None)
        global hparams
        self.hparams = copy.copy(hparams) if hp is None else copy.copy(hp)
        hparams = self.hparams

        self.z_dim = hparams['z_dim']
        self.camera_dim = 25
        self.w_dim=hparams['w_dim']

        self.img_resolution = hparams['final_resolution']
        self.img_channels = 3
        self.renderer = ImportanceRenderer(hp=hparams)
        self.ray_sampler = RaySampler()

        self.neural_rendering_resolution = hparams['neural_rendering_resolution']

        self.img2plane_backbone = Img2PlaneModel()

        self.decoder = OSGDecoder(32, {'decoder_lr_mul': 1, 'decoder_output_dim': 32})
        
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
                            # 'ray_start': hparams['ray_near'], 'ray_end': hparams['ray_far'],
                            'box_warp': 1., # 3DMM坐标系==world坐标系，而3DMM的landmark的坐标均位于[-1,1]内
                            'avg_camera_radius': 2.7,
                            'avg_camera_pivot': [0, 0, 0.2],
                            'white_back': False,
                            }
        
        sr_num_fp16_res = hparams['num_fp16_layers_in_super_resolution']
        sr_kwargs = {'channel_base': hparams['base_channel'], 'channel_max': hparams['max_channel'], 'fused_modconv_default': 'inference_only'}
        self.superresolution = SuperresolutionHybrid8XDC(channels=32, img_resolution=self.img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=True, **sr_kwargs)

    def cal_plane(self, img):
        planes = self.img2plane_backbone.forward(img)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return planes
    
    def synthesis(self, img, camera, cond=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = camera[:, :16].view(-1, 4, 4)
        intrinsics = camera[:, 16:25].view(-1, 3, 3)

        neural_rendering_resolution = self.neural_rendering_resolution

        # Create a batch of rays for volume rendering
        # ray_origins, ray_directions = self.ray_sampler.forward_with_src_c2w(ref_cam2world_matrix, cam2world_matrix, intrinsics, neural_rendering_resolution)
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.img2plane_backbone.forward(img)
        if cache_backbone:
            self._last_planes = planes
        
        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1]) # [B, 3, 32, W, H]

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, _ = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        ws_to_sr = torch.ones([feature_image.shape[0], 14, hparams['w_dim']], dtype=feature_image.dtype, device=feature_image.device)
        sr_image = self.superresolution(rgb_image, feature_image, ws_to_sr, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        ret = {'image_raw': rgb_image, 'image_depth': depth_image, 'image': sr_image, 'image_feature': feature_image[:, 3:], 'plane': planes}
        return ret
    
    def sample(self, coordinates, directions, img, cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        planes = self.img2plane_backbone.forward(img, cond=cond)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, img, camera, cond=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, return_all=True, **synthesis_kwargs):
        # Render a batch of generated images.
        out = self.synthesis(img, camera, cond=cond, update_emas=update_emas, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        return out


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions=None, **kwargs):
        # Aggregate features
        if sampled_features.shape[1] == 3:
            sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.reshape(N*M, C)

        x = self.net(x)
        x = x.reshape(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}
