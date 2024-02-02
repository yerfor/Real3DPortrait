import torch
from modules.real3d.secc_img2plane import OSAvatarSECC_Img2plane
from modules.real3d.super_resolution.sr_with_ref import SuperresolutionHybrid8XDC_Warp
from utils.commons.hparams import hparams


class OSAvatarSECC_Img2plane_Torso(OSAvatarSECC_Img2plane):
    def __init__(self, hp=None):
        super().__init__(hp=hp)
        del self.superresolution
        self.superresolution = SuperresolutionHybrid8XDC_Warp(channels=32, img_resolution=self.img_resolution, sr_num_fp16_res=self.sr_num_fp16_res, sr_antialias=True, **self.sr_kwargs)
    
    def _forward_sr(self, rgb_image, feature_image, cond, ret, **synthesis_kwargs):
        hparams = self.hparams
        ones_ws = torch.ones([feature_image.shape[0], 14, hparams['w_dim']], dtype=feature_image.dtype, device=feature_image.device)
        sr_image, facev2v_ret = self.superresolution(rgb_image, feature_image, ones_ws, cond['ref_torso_img'], cond['bg_img'], ret['weights_img'], cond['segmap'], cond['kp_s'], cond['kp_d'], cond.get('target_torso_mask'), noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        ret.update(facev2v_ret)        
        return sr_image

    def infer_synthesis_stage1(self, img, camera, cond=None, ret=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
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
            planes = self.cal_plane(img, cond)
        if cache_backbone:
            self._last_planes = planes
        
        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1]) # [B, 3, 32, W, H]

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
        ones_ws = torch.ones([feature_image.shape[0], 14, hparams['w_dim']], dtype=feature_image.dtype, device=feature_image.device)
        facev2v_ret = self.superresolution.infer_forward_stage1(rgb_image, feature_image, ones_ws, cond['ref_torso_img'], cond['bg_img'], ret['weights_img'], cond['segmap'], cond['kp_s'], cond['kp_d'], noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        rgb_image = rgb_image.clamp(-1,1)
        facev2v_ret.update({'image_raw': rgb_image, 'image_depth': depth_image, 'image_feature': feature_image[:, 3:], 'plane': planes})
        return facev2v_ret
    
    def infer_synthesis_stage2(self, facev2v_ret, **synthesis_kwargs):
        hparams = self.hparams
        ret = facev2v_ret
        sr_image, facev2v_ret = self.superresolution.infer_forward_stage2(facev2v_ret, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        sr_image = sr_image.clamp(-1,1)
        facev2v_ret['image'] = sr_image
        return ret