import numpy as np
import torch
import torch.distributed as dist
import os
import copy
import cv2
import hashlib
import tqdm
import scipy
import uuid
import random
from PIL import Image

from utils.commons.hparams import hparams
from utils.commons.tensor_utils import tensors_to_scalars, convert_to_np, move_to_cuda
from utils.nn.model_utils import not_requires_grad, num_params
from utils.nn.schedulers import NoneSchedule
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, restore_weights, restore_opt_state

from utils.commons.base_task import BaseTask
from tasks.os_avatar.loss_utils.vgg19_loss import VGG19Loss
from modules.real3d.facev2v_warp.losses import PerceptualLoss
from tasks.os_avatar.dataset_utils.motion2video_dataset import Img2Plane_Dataset
import lpips
from utils.commons.dataset_utils import data_loader

from modules.real3d.img2plane_baseline import OSAvatar_Img2plane
from modules.eg3ds.models.triplane import TriPlaneGenerator
from modules.eg3ds.models.dual_discriminator import DualDiscriminator
from modules.eg3ds.torch_utils.ops import conv2d_gradfix
from modules.eg3ds.torch_utils.ops import upfirdn2d
from modules.eg3ds.models.dual_discriminator import filtered_resizing


class ScheduleForImg2Plane(NoneSchedule):
    def __init__(self, optimizer, lr, lr_d, warmup_updates=0):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr
        self.lr_d = lr_d
        self.warmup_updates = warmup_updates
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            warmup = min(num_updates / self.warmup_updates, 1.0)
            self.lr = max(constant_lr * warmup, 1e-7)
        else:
            self.lr = constant_lr

        for optim_i in range(len(self.optimizer)-1):
            self.optimizer[optim_i].param_groups[0]['lr'] = max(1e-5, self.lr * (hparams.get("lr_decay_rate", 0.95)) ** (num_updates // hparams.get("lr_decay_interval", 5_000))) # secc_img2plane
            self.optimizer[optim_i].param_groups[1]['lr'] = max(1e-5, self.lr * (hparams.get("lr_decay_rate", 0.95)) ** (num_updates // hparams.get("lr_decay_interval", 5_000))) if num_updates >= min(2_000, hparams['start_adv_iters']) else 0 # decoder
            # fix住来自预训练EG3D的超分, 给img2plane 30000步的warmup时间
            self.optimizer[optim_i].param_groups[2]['lr'] = max(1e-5, self.lr * (hparams.get("lr_decay_rate", 0.95)) ** (num_updates // hparams.get("lr_decay_interval", 5_000)))  if num_updates >= hparams['start_adv_iters'] else 0 # SR module

        self.optimizer[-1].param_groups[0]['lr'] = self.lr_d * 1 # for disc
        return self.lr
    

class OSAvatarImg2PlaneTask(BaseTask):
    def __init__(self):
        super().__init__()
        if hparams['lpips_mode'] == 'vgg19':
            self.criterion_lpips = VGG19Loss()
        elif hparams['lpips_mode'] in ['vgg16', 'vgg']: 
            hparams['lpips_mode'] = 'vgg'
            self.criterion_lpips = lpips.LPIPS(net=hparams['lpips_mode'],lpips=True)
        elif hparams['lpips_mode'] == 'vgg19_v2':
            self.criterion_lpips = PerceptualLoss()
        else:
            raise NotImplementedError
        self.gen_tmp_output = {}
        if hparams.get("use_kv_dataset", True):
            self.dataset_cls = Img2Plane_Dataset
        else:
            raise NotImplementedError()
        self.start_adv_iters = hparams["start_adv_iters"]
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1])
    
    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(prefix='train')
        self.train_dl = train_dataset.get_dataloader()
        return self.train_dl

    @data_loader
    def val_dataloader(self):
        val_dataset = self.dataset_cls(prefix='val')
        self.val_dl = val_dataset.get_dataloader()
        return self.val_dl

    @data_loader
    def test_dataloader(self):
        val_dataset = self.dataset_cls(prefix='val')
        self.val_dl = val_dataset.get_dataloader()
        return self.val_dl
    
    def build_model(self):
        self.eg3d_model = TriPlaneGenerator()
        load_ckpt(self.eg3d_model, hparams['pretrained_eg3d_ckpt'], strict=True)
        self.model = OSAvatar_Img2plane()
        load_ckpt(self.model, hparams['pretrained_eg3d_ckpt'], strict=False, verbose=False)
        self.disc = DualDiscriminator()
        load_ckpt(self.disc, hparams['pretrained_eg3d_ckpt'], strict=False, model_name='disc')
        if hparams.get('init_from_ckpt', '') != '':
            # load_ckpt(self.model.img2plane_backbone, ckpt_dir, model_name='model.img2plane_backbone', strict=True)
            load_ckpt(self.model, hparams['init_from_ckpt'], model_name='model', strict=False)
            # load_ckpt(self.model, hparams['init_from_ckpt'], model_name='model', strict=True)
            load_ckpt(self.disc, hparams['init_from_ckpt'], model_name='disc', strict=True)
            # restore_weights(self, get_last_checkpoint(hparams.get('init_from_ckpt', ''))[0])
            print(f"restored weights from {hparams.get('init_from_ckpt', '')}")

        # define groups of params, used to assign optimizer.
        self.img2plane_backbone_params = [p for k, p in self.model.img2plane_backbone.named_parameters() if p.requires_grad]
        self.decoder_params = [p for p in self.model.decoder.parameters() if p.requires_grad]
        self.upsample_params = [p for p in self.model.superresolution.parameters() if p.requires_grad]
        self.disc_params = [p for k, p in self.disc.named_parameters() if p.requires_grad]
        return self.model
    
    def build_optimizer(self, model):
        self.optimizer_gen = optimizer_gen = torch.optim.Adam(
            self.img2plane_backbone_params,
            lr=hparams['lr_g'], 
            betas=(hparams['optimizer_adam_beta1_g'], hparams['optimizer_adam_beta2_g'])
        )
        self.optimizer_gen.add_param_group({
            'params': self.decoder_params,
            'lr': hparams['lr_g'],
            'betas': (hparams['optimizer_adam_beta1_g'], hparams['optimizer_adam_beta2_g'])
        })
        self.optimizer_gen.add_param_group({
            'params': self.upsample_params,
            'lr': hparams['lr_g'],
            'betas': (hparams['optimizer_adam_beta1_g'], hparams['optimizer_adam_beta2_g'])
        })

        mb_ratio_d = hparams['reg_interval_d'] / (hparams['reg_interval_d']  + 1)
        self.optimizer_disc = optimizer_disc = torch.optim.Adam(
            self.disc_params,
            lr=hparams['lr_d'] * mb_ratio_d,
            betas=(hparams['optimizer_adam_beta1_d'] ** mb_ratio_d, hparams['optimizer_adam_beta2_d'] ** mb_ratio_d))
        optimizers = [optimizer_gen] * 2 + [optimizer_disc] # optim0-1: ref/mv; optim2: disc
        return optimizers
    
    def build_scheduler(self, optimizer):
        mb_ratio_d = hparams['reg_interval_d'] / (hparams['reg_interval_d']  + 1)
        return ScheduleForImg2Plane(optimizer, hparams['lr_g'], hparams['lr_d'] * mb_ratio_d, hparams['warmup_updates'])

    def on_train_start(self):
        print("==============================")
        num_params(self.model, model_name="Generator")
        for n, m in self.model.named_children():
            num_params(m, model_name="|-- "+n)
        print("==============================")
        num_params(self.disc, model_name="Discriminator")
        for n, m in self.disc.named_children():
            num_params(m, model_name="|-- "+n)
        print("==============================")

    def forward_G(self, img, camera, cond=None, ret=None, update_emas=False, cache_backbone=True, use_cached_backbone=False):
        """
        ref_img: [B, 3, W, H]
        camera: [b, 25], 16 dim c2w, and 9 dim intrinsic
        """
        G = self.model
        gen_output = G.forward(img=img, camera=camera, cond=cond, ret=ret, update_emas=update_emas, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone)
        return gen_output

    def forward_D(self, img, camera, update_emas=False):
        D = self.disc
        logits = D.forward(img, camera, update_emas=update_emas)
        return logits

    def prepare_batch(self, batch):
        # get synthesized img as GT data
        out_batch = {}
        ws_camera = batch['ffhq_ws_cameras']
        camera = batch['ffhq_ref_cameras']
        fake_camera = batch['ffhq_mv_cameras']
        z = torch.randn([camera.shape[0], hparams['z_dim']],device=camera.device)
        with torch.no_grad():
            ws = self.eg3d_model.mapping(z, ws_camera, update_emas=False, truncation_psi=0.7)
            ref_img_gen = self.eg3d_model.synthesis(ws, camera, cache_backbone=True, use_cached_backbone=False)
            mv_img_gen = self.eg3d_model.synthesis(ws, fake_camera, cache_backbone=False, use_cached_backbone=True)
            ref_img_gen['image_raw'] = filtered_resizing(ref_img_gen['image'], size=hparams['neural_rendering_resolution'], f=self.resample_filter, filter_mode='antialiased')
            mv_img_gen['image_raw'] = filtered_resizing(mv_img_gen['image'], size=hparams['neural_rendering_resolution'], f=self.resample_filter, filter_mode='antialiased')
        
        out_batch.update({
            'ffhq_planes': ref_img_gen['plane'],
            'ffhq_ref_imgs_raw': ref_img_gen['image_raw'],
            'ffhq_ref_imgs': ref_img_gen['image'],
            'ffhq_ref_imgs_depth': ref_img_gen['image_depth'],

            'ffhq_mv_imgs_raw': mv_img_gen['image_raw'],
            'ffhq_mv_imgs': mv_img_gen['image'],
            'ffhq_mv_imgs_depth': mv_img_gen['image_depth'],
            'ffhq_mv_imgs_feature': mv_img_gen['image_feature'],
            'ffhq_ref_cameras': batch['ffhq_ref_cameras'],
            'ffhq_mv_cameras': batch['ffhq_mv_cameras'],
        })

        return out_batch
    
    def run_G_reference_image(self, batch):
        losses = {}
        ret = {}
        if self.global_step+1 < self.start_adv_iters:
            # at early stage, don't train on the ref camera too frequently, to prevent bill-board
            if (self.global_step+1) % 5 != 0:
                return losses
        elif self.global_step < self.start_adv_iters + 5000:
            if (self.global_step+1) % 2 != 0:
                return losses
        else:
            # after 37,500 steps, we train on ref image every step.
            pass

        with torch.autograd.profiler.record_function('G_ref_forward'):
            camera = batch['ffhq_ref_cameras']
            img = batch['ffhq_ref_imgs']
            img_raw = batch['ffhq_ref_imgs_raw']
            img_depth = batch['ffhq_ref_imgs_depth']

            gen_img = self.forward_G(img, camera, cond={'ref_cameras': batch['ffhq_ref_cameras']}, ret=ret)
            if 'losses' in ret: losses.update(ret['losses'])
            self.gen_tmp_output['recon_ref_imgs'] = gen_img['image'].detach()
            self.gen_tmp_output['recon_ref_imgs_raw'] = gen_img['image_raw'].detach()

            losses['G_ref_plane_l1_mean'] = (gen_img['plane'][:,:]).detach().abs().mean()
            losses['G_ref_plane_l1_std'] = (gen_img['plane'][:,:]).detach().abs().std()
            if hparams['use_mse']:
                losses['G_ref_img_mae'] = (gen_img['image'] - img).pow(2).mean()
                losses['G_ref_img_mae_raw'] = (gen_img['image_raw'] - img_raw).pow(2).mean()
            else:
                losses['G_ref_img_mae'] = (gen_img['image'] - img).abs().mean()
                losses['G_ref_img_mae_raw'] = (gen_img['image_raw'] - img_raw).abs().mean()

            pred_img_01 = ((gen_img['image'] + 1) / 2).clamp(0, 1)
            pred_img_01_raw = ((gen_img['image_raw'] + 1) / 2).clamp(0, 1)
            gt_img_01 = ((img + 1) / 2).clamp(0, 1)
            gt_img_01_raw = ((img_raw + 1) / 2).clamp(0, 1)
            losses['G_ref_img_lpips'] = self.criterion_lpips(pred_img_01, gt_img_01).mean()
            losses['G_ref_img_lpips_raw'] = self.criterion_lpips(pred_img_01_raw, gt_img_01_raw).mean()
            losses['G_ref_img_mae_depth'] = (gen_img['image_depth'] - img_depth).detach().abs().mean()
            disc_inp_img = {
                'image': gen_img['image'],
                'image_raw': gen_img['image_raw'],
            }
            gen_logits = self.forward_D(disc_inp_img, camera)
            losses['G_ref_adv'] = torch.nn.functional.softplus(-gen_logits).mean()
        return losses

    def run_G_multiview_image(self, batch):
        losses = {}
        ret = {}
        with torch.autograd.profiler.record_function('G_mv_forward'):
            camera = batch['ffhq_mv_cameras']
            img = batch['ffhq_mv_imgs']
            img_raw = batch['ffhq_mv_imgs_raw']
            img_depth = batch['ffhq_mv_imgs_depth']

            gen_img = self.forward_G(batch['ffhq_ref_imgs'], camera, cond={'ref_cameras': batch['ffhq_ref_cameras']}, ret=ret)
            if 'losses' in ret: losses.update(ret['losses'])
            self.gen_tmp_output['recon_mv_imgs'] = gen_img['image'].detach()
            self.gen_tmp_output['recon_mv_imgs_raw'] = gen_img['image_raw'].detach()
            losses['G_mv_plane_l1_mean'] = (gen_img['plane'][:,:]).detach().abs().mean()
            losses['G_mv_plane_l1_std'] = (gen_img['plane'][:,:]).detach().abs().std()
            if hparams['use_mse']:
                losses['G_mv_img_mae'] = (gen_img['image'] - img).pow(2).mean()
                losses['G_mv_img_mae_raw'] = (gen_img['image_raw'] - img_raw).pow(2).mean()
            else:
                losses['G_mv_img_mae'] = (gen_img['image'] - img).abs().mean()
                losses['G_mv_img_mae_raw'] = (gen_img['image_raw'] - img_raw).abs().mean()
            pred_img_01 = (gen_img['image'] + 1) / 2
            pred_img_01_raw = (gen_img['image_raw'] + 1) / 2
            gt_img_01 = (img + 1) / 2
            gt_img_01_raw = (img_raw + 1) / 2
            losses['G_mv_img_lpips'] = self.criterion_lpips(pred_img_01, gt_img_01).mean()
            losses['G_mv_img_lpips_raw'] = self.criterion_lpips(pred_img_01_raw, gt_img_01_raw).mean()
            losses['G_mv_img_mae_depth'] = (gen_img['image_depth'] - img_depth).detach().abs().mean()

            disc_inp_img = {
                'image': gen_img['image'],
                'image_raw': gen_img['image_raw'],
            }
            gen_logits = self.forward_D(disc_inp_img, camera)
            losses['G_mv_adv'] = torch.nn.functional.softplus(-gen_logits).mean()
        return losses
    

    def run_G_reg(self, batch):
        losses = {}
        imgs = batch['ffhq_mv_imgs']

        if (self.global_step+1) % hparams['reg_interval_g'] == 0:
            with torch.autograd.profiler.record_function('G_regularize_forward'):
                initial_coordinates = torch.rand((imgs.shape[0], 1000, 3), device=imgs.device) - 0.5 # [-0.5,0.5]
                perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * 5e-3
                all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
                source_sigma = self.model.sample(coordinates=all_coordinates, directions=torch.randn_like(all_coordinates), img=imgs, cond={'ref_cameras': batch['ffhq_mv_cameras']}, update_emas=False)['sigma']
                source_sigma_initial = source_sigma[:, :source_sigma.shape[1]//2]
                source_sigma_perturbed = source_sigma[:, source_sigma.shape[1]//2:]
                density_reg_loss = torch.nn.functional.l1_loss(source_sigma_initial, source_sigma_perturbed)

                # we want the pertubed position has similar density
                losses['G_regularize_density_l1'] = density_reg_loss
        return losses

    def forward_D_main(self, batch):
        """
        we update ema this substep.
        """
        losses = {}
        with torch.autograd.profiler.record_function('D_minimize_fake_forward'):
            if 'recon_ref_imgs' in self.gen_tmp_output:
                camera = batch['ffhq_ref_cameras']
                disc_inp_img = {
                    'image': self.gen_tmp_output['recon_ref_imgs'],
                    'image_raw': self.gen_tmp_output['recon_ref_imgs_raw'],
                }
                gen_logits = self.forward_D(disc_inp_img, camera, update_emas=True)
                losses['D_minimize_ref_fake'] = torch.nn.functional.softplus(gen_logits).mean()

            if 'recon_mv_imgs' in self.gen_tmp_output:
                camera = batch['ffhq_mv_cameras']
                disc_inp_img = {
                    'image': self.gen_tmp_output['recon_mv_imgs'],
                    'image_raw': self.gen_tmp_output['recon_mv_imgs_raw'],
                }
                gen_logits = self.forward_D(disc_inp_img, camera, update_emas=True)
                losses['D_minimize_mv_fake'] = torch.nn.functional.softplus(gen_logits).mean()

        # Maximize confidence on true samples
        with torch.autograd.profiler.record_function('D_maximize_true_forward'):
            if hparams.get("ffhq_disc_inp_mode", "eg3d_gen") == 'eg3d_gen': 
                ref_cameras = batch['ffhq_ref_cameras']
                ref_img_tmp_image = batch['ffhq_ref_imgs'].detach().requires_grad_(True)
                ref_img_tmp_image_raw = batch['ffhq_ref_imgs_raw'].detach().requires_grad_(True)
            elif hparams.get("ffhq_disc_inp_mode", "eg3d_gen") == 'ffhq':
                ref_cameras = batch['ffhq_gt_cameras']
                ref_img_tmp_image = batch['ffhq_gt_imgs'].detach().requires_grad_(True)
                ref_img_tmp_image_raw = batch['ffhq_gt_imgs_raw'].detach().requires_grad_(True)

            ref_img_tmp = {'image': ref_img_tmp_image, 'image_raw': ref_img_tmp_image_raw}
            ref_logits = self.forward_D(ref_img_tmp, ref_cameras)
            losses['D_maximize_ref'] = torch.nn.functional.softplus(-ref_logits).mean()

            if hparams.get("ffhq_disc_inp_mode", "eg3d_gen") == 'eg3d_gen': 
                mv_cameras = batch['ffhq_mv_cameras']
                mv_img_tmp_image = batch['ffhq_mv_imgs'].detach().requires_grad_(True)
                mv_img_tmp_image_raw = batch['ffhq_mv_imgs_raw'].detach().requires_grad_(True)
                mv_img_tmp = {'image': mv_img_tmp_image, 'image_raw': mv_img_tmp_image_raw}
                mv_logits = self.forward_D(mv_img_tmp, mv_cameras)
                losses['D_maximize_mv'] = torch.nn.functional.softplus(-mv_logits).mean()

        if (self.global_step+1) % hparams['reg_interval_d'] == 0 and self.training is True:
            with torch.autograd.profiler.record_function('D_gradient_penalty_on_real_imgs_forward'), conv2d_gradfix.no_weight_gradients():
                ref_r1_grads = torch.autograd.grad(outputs=[ref_logits.sum()], inputs=[ref_img_tmp['image'], ref_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                ref_r1_grads_image = ref_r1_grads[0]
                ref_r1_grads_image_raw = ref_r1_grads[1]
                ref_r1_penalty_raw = ref_r1_grads_image_raw.square().sum([1,2,3]).mean()
                ref_r1_penalty_image = ref_r1_grads_image.square().sum([1,2,3]).mean()
                losses['D_ref_gradient_penalty'] = (ref_r1_penalty_image + ref_r1_penalty_raw) / 2

                if hparams.get("ffhq_disc_inp_mode", "eg3d_gen") == 'eg3d_gen': 
                    mv_r1_grads = torch.autograd.grad(outputs=[mv_logits.sum()], inputs=[mv_img_tmp['image'], mv_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                    mv_r1_grads_image = mv_r1_grads[0]
                    mv_r1_grads_image_raw = mv_r1_grads[1]
                    mv_r1_penalty_raw = mv_r1_grads_image_raw.square().sum([1,2,3]).mean()
                    mv_r1_penalty_image = mv_r1_grads_image.square().sum([1,2,3]).mean()
                    losses['D_mv_gradient_penalty'] = (mv_r1_penalty_image + mv_r1_penalty_raw) / 2

        self.gen_tmp_output = {}
        return losses

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if len(sample) == 0:
            return None 
        if optimizer_idx == 0:
            sample = self.prepare_batch(sample)
            self.cache_sample = sample
        else:
            sample = self.cache_sample
        # self.update_ema()
        losses = {}
        if optimizer_idx == 0:
            losses.update(self.run_G_reference_image(sample))
            # after the early stage, we decrease the lambda of error-based loss to prevent oversmoothness
            loss_weights = {
                'G_ref_img_mae': 1.0,
                'G_ref_img_mae_raw': 1.0,
                'G_ref_img_mae_depth': hparams.get('lambda_mse_depth', 0),
                'G_ref_img_lpips': 0.1,
                'G_ref_img_lpips_raw': 0.1,
                'G_ref_adv': 0.0 if self.global_step < self.start_adv_iters else 0.1,
            }

        elif optimizer_idx == 1:
            losses.update(self.run_G_multiview_image(sample))
            losses.update(self.run_G_reg(sample))
            loss_weights = {
                'G_mv_img_mae': 1.0,
                'G_mv_img_mae_raw': 1.0,
                'G_mv_img_mae_depth': hparams.get('lambda_mse_depth', 0),
                'G_mv_img_lpips': 0.1,
                'G_mv_img_lpips_raw': 0.1,
                'G_mv_adv': 0.0 if self.global_step < self.start_adv_iters else 0.025,
                'G_regularize_density_l1': hparams['lambda_density_reg'],
            }

        elif optimizer_idx == 2:
            losses.update(self.forward_D_main(sample))
            loss_weights = {
                'D_maximize_ref': 1.0,
                'D_minimize_ref_fake': 1.0,
                'D_ref_gradient_penalty': hparams['lambda_gradient_penalty'] * hparams['reg_interval_d'],
                'D_maximize_mv': 1.0,
                'D_minimize_mv_fake': 1.0,
                'D_mv_gradient_penalty': hparams['lambda_gradient_penalty'] * hparams['reg_interval_d'],
            }
        total_loss = sum([loss_weights[k] * v for k, v in losses.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        if len(losses) == 0:
            return None
        return total_loss, losses
    
    #####################
    # Validation
    #####################
    def validation_start(self):
        if self.global_step % hparams['valid_infer_interval'] == 0:
            self.gen_dir = os.path.join(hparams['work_dir'], f'validation_results')
            os.makedirs(self.gen_dir, exist_ok=True)

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        losses = {}
        if len(sample) == 0:
            return None

        sample = self.prepare_batch(sample)
        rank = 0 if len(set(os.environ['CUDA_VISIBLE_DEVICES'].split(","))) == 1 else dist.get_rank()

        losses.update(self.run_G_reference_image(sample))
        losses.update(self.run_G_multiview_image(sample))
        losses.update(self.forward_D_main(sample))
        outputs['losses'] = losses
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = tensors_to_scalars(outputs)

        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots'] and rank == 0:

            imgs_ref = sample['ffhq_ref_imgs']
            dummy_cond = {
                'cond_src': torch.zeros([imgs_ref.shape[0], 3, 512, 512], dtype=sample['ffhq_mv_cameras'].dtype, device=sample['ffhq_mv_cameras'].device),
                'cond_drv': torch.zeros([imgs_ref.shape[0], 3, 512, 512], dtype=sample['ffhq_mv_cameras'].dtype, device=sample['ffhq_mv_cameras'].device),
                'ref_cameras': sample['ffhq_ref_cameras'],
            }
            gen_img = self.model.forward(imgs_ref, sample['ffhq_mv_cameras'], cond=dummy_cond, noise_mode='const')
            gen_img_recon = self.model.forward(imgs_ref, sample['ffhq_ref_cameras'], cond=dummy_cond, noise_mode='const')
            imgs_recon = gen_img_recon['image'].permute(0,2,3,1)
            imgs_recon_raw = filtered_resizing(gen_img_recon['image_raw'], size=512, f=self.resample_filter, filter_mode='antialiased').permute(0, 2,3,1)
            imgs_recon_depth = gen_img_recon['image_depth'].permute(0,2,3,1)
            imgs_pred = gen_img['image'].permute(0,2,3,1)
            imgs_pred_raw = filtered_resizing(gen_img['image_raw'], size=512, f=self.resample_filter, filter_mode='antialiased').permute(0, 2,3,1)
            imgs_pred_depth = gen_img['image_depth'].permute(0,2,3,1)
            imgs_ref = imgs_ref.permute(0,2,3,1)
            imgs_mv = sample['ffhq_mv_imgs'].permute(0,2,3,1) # [B, H, W, 3]

            for i in range(len(imgs_pred)):
                idx_string = format(i+batch_idx * hparams['batch_size'], "05d")
                base_fn = f"{idx_string}"
                img_ref_mv_recon_pred = torch.cat([imgs_ref[i], imgs_mv[i], imgs_recon_raw[i], imgs_pred_raw[i], imgs_recon[i], imgs_pred[i]], dim=1)
                self.save_rgb_to_fname(img_ref_mv_recon_pred, f"{self.gen_dir}/images_rgb_iter{self.global_step}/ref_mv_recon_pred_{base_fn}.png")
                img_depth_recon_pred = torch.cat([imgs_recon_depth[i], imgs_pred_depth[i]], dim=1)
                self.save_depth_to_fname(img_depth_recon_pred, f"{self.gen_dir}/images_depth_iter{self.global_step}/recon_pred_{base_fn}.png") 
            
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)
    
    @staticmethod
    def save_rgb_to_fname(rgb, fname):
        """
        rgb: [H, W, 3]
        """
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        img = (rgb * 127.5 + 128).clamp(0, 255)
        img = convert_to_np(img).astype(np.uint8)

        Image.fromarray(img, 'RGB').save(fname)

    @staticmethod
    def save_depth_to_fname(depth, fname):
        """
        depth: [H, W, 3]
        """
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        low, high = depth.min(), depth.max()
        img = (depth - low) * (255 / (high - low))
        img = convert_to_np(img)
        img = np.rint(img).clip(0, 255).astype(np.uint8)

        Image.fromarray(img[:, :, 0], 'L').save(fname)
