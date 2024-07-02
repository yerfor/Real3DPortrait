# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torch.nn as nn
from modules.eg3ds.models.networks_stylegan2 import Generator as StyleGAN2Backbone
from modules.eg3ds.models.networks_stylegan2 import FullyConnectedLayer
from modules.eg3ds.volumetric_rendering.renderer import ImportanceRenderer
from modules.eg3ds.volumetric_rendering.ray_sampler import RaySampler
from modules.eg3ds.models.superresolution import SuperresolutionHybrid2X, SuperresolutionHybrid4X, SuperresolutionHybrid8X, SuperresolutionHybrid8XDC

import copy
from utils.commons.hparams import hparams


class TriPlaneGenerator(torch.nn.Module):
    def __init__(self, hp=None):
        super().__init__()
        global hparams
        self.hparams = copy.copy(hparams) if hp is None else copy.copy(hp)
        hparams = self.hparams

        self.z_dim = hparams['z_dim']
        self.camera_dim = 25
        self.w_dim=hparams['w_dim']

        self.img_resolution = hparams['final_resolution']
        self.img_channels = 3
        self.renderer = ImportanceRenderer(hp=hparams)
        self.renderer.triplane_feature_type = 'triplane'
        self.ray_sampler = RaySampler()

        self.neural_rendering_resolution = hparams['neural_rendering_resolution']

        mapping_kwargs = {'num_layers': hparams['mapping_network_depth']}
        synthesis_kwargs = {'channel_base': hparams['base_channel'], 'channel_max': hparams['max_channel'], 'fused_modconv_default': 'inference_only', 'num_fp16_res': hparams['num_fp16_layers_in_generator'], 'conv_clamp': None}

        triplane_c_dim = self.camera_dim        

        # if gen_cond_mode == 'mapping', add a cond_mapping in backbone
        self.backbone = StyleGAN2Backbone(self.z_dim, triplane_c_dim, self.w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
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
                            'ray_start': hparams['ray_near'], 'ray_end': hparams['ray_far'],
                            'box_warp': hparams['box_warp'], 
                            'avg_camera_radius': 2.7, # 仅仅用在infer的pose sampler里面，在那里相机围绕一个半径恒定的球移动，这个半径代表着camera距离世界坐标系中心的距离。
                            'avg_camera_pivot': [0, 0, 0.2], # 仅仅用在infer的pose sampler里面，代表着camera看向的位置，这决定了view direction。这里的[0.,0.,0.2]应该是3dmm人脸的“人中”
                            'white_back': False, # 如果背景是纯白色可以考虑启用，因为默认无density的世界是黑色的，这个设置让默认世界变成白色，这让网络不需要建模一层薄薄的voxel来生成白色背景。
                            }
        
        sr_num_fp16_res = hparams['num_fp16_layers_in_super_resolution']
        sr_kwargs = {'channel_base': hparams['base_channel'], 'channel_max': hparams['max_channel'], 'fused_modconv_default': 'inference_only'}
        self.superresolution = SuperresolutionHybrid8XDC(channels=32, img_resolution=self.img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=True, **sr_kwargs)

    def mapping(self, z, camera, cond=None, truncation_psi=0.7, truncation_cutoff=None, update_emas=False):
        """
        Generate weights by forward the Mapping network.

        z: latent sampled from N(0,1): [B, z_dim=512]
        camera: falttened extrinsic 4x4 matrix and intrinsic 3x3 matrix [B, c=16+9]
        cond: auxiliary condition, such as idexp_lm3d: [B, c=68*3]
        truncation_psi: the threshold of truncation trick in BigGAN, 1.0 means no effect, 0.0 means the ws is the mean_ws, and 0~1 value means linear interpolation in these two.
        truncation_cutoff: number of ws to adopt truncation. default None means adopt to all ws. other int mean the first number of layers to adopt this trick.
        """
        c = camera
        ws = self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        if hparams.get("gen_cond_mode", 'none') == 'mapping':
            d_ws = self.backbone.cond_mapping(cond, None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            ws = ws * 0.5 + d_ws * 0.5
        return ws
            
    def synthesis(self, ws, camera, cond=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        """
        Run the Backbone to synthesize images given the ws generated by self.mapping
        """
        ret = {}

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
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, -1, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, is_ray_valid = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        if hparams.get("mask_invalid_rays", False):
            is_ray_valid_mask = is_ray_valid.reshape([feature_samples.shape[0], 1,self.neural_rendering_resolution,self.neural_rendering_resolution]) # [B, 1, H, W]
            feature_image[~is_ray_valid_mask.repeat([1,feature_image.shape[1],1,1])] = -1
            depth_image[~is_ray_valid_mask] = depth_image[is_ray_valid_mask].min().item()

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        ws_to_sr = ws
        if hparams['ones_ws_for_sr']:
            ws_to_sr = torch.ones_like(ws)
        sr_image = self.superresolution(rgb_image, feature_image, ws_to_sr, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        rgb_image = rgb_image.clamp(-1,1)
        sr_image = sr_image.clamp(-1,1)
        ret.update({'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_feature': feature_image[:, 3:], 'plane': planes})
        return ret
    
    def sample(self, coordinates, directions, z, camera, cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        """
        Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        Not aggregated into pixels, but in the world coordinate.
        """
        ws = self.mapping(z, camera, cond=cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        """
        Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        """
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, camera, cond=None, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        """
        Render a batch of generated images.
        """
        ws = self.mapping(z, camera, cond=cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, camera, cond=cond, update_emas=update_emas, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}
