import numpy as np
import torch
import torch.distributed as dist
import os
import random 
import cv2

from utils.commons.hparams import hparams
from utils.commons.tensor_utils import tensors_to_scalars, convert_to_np, move_to_cuda
from utils.nn.model_utils import not_requires_grad, num_params
from utils.commons.dataset_utils import data_loader
from utils.nn.schedulers import NoneSchedule
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, restore_weights, restore_opt_state

from data_util.face3d_helper import Face3DHelper
from deep_3drecon.secc_renderer import SECC_Renderer
from tasks.os_avatar.loss_utils.vgg19_loss import VGG19Loss
from tasks.os_avatar.secc_img2plane_task import SECC_Img2PlaneEG3DTask
import lpips

from tasks.os_avatar.dataset_utils.motion2video_dataset import Motion2Video_Dataset

from modules.eg3ds.models.triplane import TriPlaneGenerator
from modules.eg3ds.models.dual_discriminator import DualDiscriminator
from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane_Torso

from modules.eg3ds.torch_utils.ops import conv2d_gradfix
from modules.eg3ds.torch_utils.ops import upfirdn2d
from modules.eg3ds.models.dual_discriminator import filtered_resizing


class ScheduleForLM3DImg2PlaneEG3D(NoneSchedule):
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
            self.optimizer[optim_i].param_groups[0]['lr'] = max(1e-6, self.lr * (0.5) ** (num_updates // 50_000))
        self.optimizer[-1].param_groups[0]['lr'] = max(1e-6, self.lr_d * (0.5) ** (num_updates // 50_000)) # for disc
        return self.lr
    

class SECC_Img2PlaneEG3D_TorsoTask(SECC_Img2PlaneEG3DTask):
    def build_model(self):
        self.eg3d_model = TriPlaneGenerator()
        load_ckpt(self.eg3d_model, hparams['pretrained_eg3d_ckpt'], strict=True)

        self.model = OSAvatarSECC_Img2plane_Torso()
        self.disc = DualDiscriminator()

        assert hparams.get("img2plane_backbone_mode", "composite") == "composite"
        
        assert hparams.get('init_from_ckpt', '') != '', "set init_from_ckpt with your secc2plane or secc2plane_torso ckpt!"
        ckpt_dir = hparams.get('init_from_ckpt', '')
        load_ckpt(self.model.cano_img2plane_backbone, ckpt_dir, model_name='model.cano_img2plane_backbone', strict=True)
        load_ckpt(self.model.secc_img2plane_backbone, ckpt_dir, model_name='model.secc_img2plane_backbone', strict=True)
        load_ckpt(self.model.decoder, ckpt_dir, model_name='model.decoder', strict=True)
        load_ckpt(self.model.superresolution, ckpt_dir, model_name='model.superresolution', strict=False)
        load_ckpt(self.disc, ckpt_dir, model_name='disc', strict=True)

        secc_img2plane_ckpt_dir = hparams.get('reload_head_ckpt', '')
        if secc_img2plane_ckpt_dir != '':
            load_ckpt(self.model.cano_img2plane_backbone, secc_img2plane_ckpt_dir, model_name='model.cano_img2plane_backbone', strict=True)
            load_ckpt(self.model.secc_img2plane_backbone, secc_img2plane_ckpt_dir, model_name='model.secc_img2plane_backbone', strict=True)
            load_ckpt(self.model.decoder, secc_img2plane_ckpt_dir, model_name='model.decoder', strict=True)
        
        # only update the torso-based superresolution module
        self.upsample_params = [p for p in self.model.superresolution.parameters() if p.requires_grad]
        self.disc_params = [p for k, p in self.disc.named_parameters() if p.requires_grad] 

        if hparams.get("add_ffhq_singe_disc", False):
            self.ffhq_disc = DualDiscriminator()
            self.disc_params += [p for k, p in self.ffhq_disc.named_parameters() if p.requires_grad] 
            eg3d_dir = 'checkpoints/geneface2_ckpts/eg3d_baseline_run2'
            load_ckpt(self.ffhq_disc, eg3d_dir, model_name='disc', strict=True)

        self.secc_renderer = SECC_Renderer(512)
        self.face3d_helper = Face3DHelper(use_gpu=False)
        return self.model

    def on_train_start(self):
        print("==============================")
        num_params(self.model, model_name="Generator")
        for n, m in self.model.named_children():
            num_params(m, model_name="|-- "+n)
        print("==============================")
        for n, m in self.model.superresolution.named_children():
            num_params(m, model_name="|-- "+ "SR module --"+n)
        print("==============================")
        num_params(self.disc, model_name="Discriminator")
        for n, m in self.disc.named_children():
            num_params(m, model_name="|-- "+n)
        print("==============================")

    def build_optimizer(self, model):
        self.optimizer_gen = optimizer_gen = torch.optim.Adam(
            self.upsample_params,
            lr=hparams['lr_g'], # we use a 0.5x smaller lr for transformer
            betas=(hparams['optimizer_adam_beta1_g'], hparams['optimizer_adam_beta2_g'])
        )

        mb_ratio_d = hparams['reg_interval_d'] / (hparams['reg_interval_d']  + 1)
        self.optimizer_disc = optimizer_disc = torch.optim.Adam(
            self.disc_params,
            lr=hparams['lr_d'] * mb_ratio_d,
            betas=(hparams['optimizer_adam_beta1_d'] ** mb_ratio_d, hparams['optimizer_adam_beta2_d'] ** mb_ratio_d))
        optimizers = [optimizer_gen, optimizer_disc]
        return optimizers
    
    def build_scheduler(self, optimizer):
        mb_ratio_d = hparams['reg_interval_d'] / (hparams['reg_interval_d']  + 1)
        return ScheduleForLM3DImg2PlaneEG3D(optimizer, hparams['lr_g'], hparams['lr_d'] * mb_ratio_d, hparams['warmup_updates'])
    
    def prepare_batch(self, batch):
        out_batch = super().prepare_batch(batch)

        if hparams.get("add_ffhq_singe_disc", False) and (self.global_step+1) % 4 == 0:
            batch_size = batch['th1kh_ref_cameras'].shape[0]
            ffhq_img_lst = []
            ffhq_head_img_dir = '/mnt/bn/sa-ag-data/yezhenhui/datasets/raw/FFHQ/com_imgs'
            while len(ffhq_img_lst) < batch_size:
                idx = random.randint(0, 70000-1)
                img_name = f"{ffhq_head_img_dir}/{format(idx,'05d')}.png"
                if os.path.exists(img_name):
                    img = cv2.imread(img_name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 127.5 - 1
                    img = torch.tensor(img, dtype=batch['th1kh_ref_cameras'].dtype, device=batch['th1kh_ref_cameras'].device)
                    ffhq_img_lst.append(img)
            ffhq_head_img = torch.stack(ffhq_img_lst).permute(0, 3, 1, 2)
            out_batch['ffhq_com_imgs'] = ffhq_head_img
            out_batch['ffhq_com_imgs_raw'] = filtered_resizing(out_batch['ffhq_head_imgs'], size=hparams['neural_rendering_resolution'], f=self.resample_filter, filter_mode='antialiased')

        out_batch['th1kh_bg_imgs'] = batch['th1kh_bg_imgs']
        out_batch['th1kh_ref_com_imgs'] = batch['th1kh_ref_com_imgs']
        out_batch['th1kh_tgt_imgs'] = batch['th1kh_mv_com_imgs']
        out_batch['th1kh_ref_segmaps'] = batch['th1kh_ref_segmaps']

        torso_ref_segout_mode = hparams.get("torso_ref_segout_mode", "torso")
        # assert torso_ref_segout_mode in ['person', 'torso', 'full', 'torso_with_bg']
        assert torso_ref_segout_mode in ['full', 'torso_with_bg', 'torso', 'person']
        if torso_ref_segout_mode == 'full':
            out_batch['th1kh_ref_torso_imgs'] = batch['th1kh_ref_com_imgs']
        elif torso_ref_segout_mode == 'torso_with_bg':
            out_batch['th1kh_ref_torso_imgs'] = batch['th1kh_ref_inpaint_torso_with_com_bg_imgs']
        elif torso_ref_segout_mode == 'torso':
            out_batch['th1kh_ref_torso_imgs'] = batch['th1kh_ref_inpaint_torso_imgs']
        elif torso_ref_segout_mode == 'person':
            out_batch['th1kh_ref_torso_imgs'] = batch['th1kh_ref_person_imgs']
        else: raise NotImplementedError()

        ref_id, ref_exp, ref_euler, ref_trans = batch['th1kh_ref_ids'], batch['th1kh_ref_exps'], batch['th1kh_ref_eulers'], batch['th1kh_ref_trans']
        ref_kp = self.face3d_helper.reconstruct_lm2d(ref_id, ref_exp, ref_euler, ref_trans)
        ref_kp = (ref_kp - 0.5) * 2 # map to -1~1
        ref_kp = torch.cat([ref_kp, torch.zeros([ref_kp.shape[0], ref_kp.shape[1], 1]).to(ref_kp.device)], dim=-1)

        mv_id, mv_exp, mv_euler, mv_trans = batch['th1kh_mv_ids'], batch['th1kh_mv_exps'], batch['th1kh_mv_eulers'], batch['th1kh_mv_trans']
        mv_kp = self.face3d_helper.reconstruct_lm2d(mv_id, mv_exp, mv_euler, mv_trans)
        mv_kp = (mv_kp - 0.5) * 2 # map to -1~1
        mv_kp = torch.cat([mv_kp, torch.zeros([mv_kp.shape[0], mv_kp.shape[1], 1]).to(mv_kp.device)], dim=-1)

        out_batch.update({
            'th1kh_ref_kp': ref_kp,
            'th1kh_mv_kp': mv_kp,
        })

        batch['th1kh_ref_torso_masks'] = self.dilate_mask(batch['th1kh_ref_torso_masks'].float(), ksize=41).long()
        out_batch['th1kh_ref_torso_masks'] = batch['th1kh_ref_torso_masks'].bool()
        out_batch['th1kh_ref_torso_masks_raw'] = torch.nn.functional.interpolate(batch['th1kh_ref_torso_masks'].unsqueeze(1).float(), size=(128,128), mode='nearest').squeeze(1).bool()
        
        batch['th1kh_mv_torso_masks'] = self.dilate_mask(batch['th1kh_mv_torso_masks'].float(), ksize=41).long()
        out_batch['th1kh_mv_torso_masks'] = batch['th1kh_mv_torso_masks'].bool()
        out_batch['th1kh_mv_torso_masks_raw'] = torch.nn.functional.interpolate(batch['th1kh_mv_torso_masks'].unsqueeze(1).float(), size=(128,128), mode='nearest').squeeze(1).bool()


        return out_batch

    def run_G_th1kh_src2src_image(self, batch):
        ret = {}
        losses = {}
        SRC2SRC_UPDATE_INTERVAL = 4
        if self.global_step % SRC2SRC_UPDATE_INTERVAL != 0:
            return losses

        with torch.autograd.profiler.record_function('G_th1kh_ref_forward'):
            camera = batch['th1kh_ref_cameras']
            img = batch['th1kh_ref_com_imgs']
            gen_img = self.forward_G(batch['th1kh_ref_head_imgs'], camera, cond={
                        'cond_cano': batch['th1kh_cano_secc'], 'cond_src': batch['th1kh_ref_secc'], 'cond_tgt': batch['th1kh_ref_secc'], 
                        'ref_torso_img': batch['th1kh_ref_torso_imgs'], 'bg_img': batch['th1kh_bg_imgs'],
                        'segmap': batch['th1kh_ref_segmaps'],
                        'kp_s':batch['th1kh_ref_kp'], 'kp_d': batch['th1kh_ref_kp'],
                        'target_torso_mask': batch['th1kh_ref_torso_masks_raw'],
                        }, ret=ret)
            if 'losses' in ret: losses.update(ret['losses'])
            
            losses['G_ref_plane_l1_mean'] = (gen_img['plane'][:,:]).detach().abs().mean()
            losses['G_ref_plane_l1_std'] = (gen_img['plane'][:,:]).detach().abs().std()
            if hparams.get("masked_error", True):
                # 之所以用L1不用MSE，原因是mse对mismatch的pixel loss过大，而导致面部细节被忽略，此外还有过模糊的问题
                # 对raw图像，因为deform的原因背景没法全黑，导致这部分mse过高，我们将其mask掉，只计算人脸部分
                losses['G_th1kh_ref_img_mae'] = self.masked_error_loss(gen_img['image'], img, batch['th1kh_ref_torso_masks'], mode='l1', unmasked_weight=0.5)
                losses['G_th1kh_ref_img_lpips'] = self.criterion_lpips(gen_img['image'], img).mean()
            else:
                losses['G_th1kh_ref_img_mae'] = (gen_img['image'] - img).abs().mean()
                losses['G_th1kh_ref_img_lpips'] = self.criterion_lpips(gen_img['image'], img).mean()
            
            # lip loss
            batch_size = len(gen_img['image'])
            lip_mse_loss = 0
            lip_lpips_loss = 0
            for i in range(batch_size):
                xmin, xmax, ymin, ymax = batch['th1kh_ref_lip_rects'][i]
                lip_tgt_imgs = img[i:i+1,:, ymin:ymax,xmin:xmax].contiguous()
                lip_pred_imgs = gen_img['image'][i:i+1,:, ymin:ymax,xmin:xmax].contiguous()
                lip_mse_loss = lip_mse_loss + 1/batch_size * (lip_pred_imgs - lip_tgt_imgs).abs().mean()
                try:
                    lip_lpips_loss = lip_lpips_loss + 1/batch_size * self.criterion_lpips(lip_pred_imgs, lip_tgt_imgs).mean()
                except: pass 
            losses['G_th1kh_ref_img_lip_mae'] = lip_mse_loss
            losses['G_th1kh_ref_img_lip_lpips'] = lip_lpips_loss

            disc_inp_img = {
                'image': gen_img['image'],
                'image_raw': gen_img['image_raw'],
            }
            gen_logits = self.forward_D(disc_inp_img, camera)
            losses['G_th1kh_ref_adv'] = torch.nn.functional.softplus(-gen_logits).mean()
            if hparams.get("add_ffhq_singe_disc", False):
                gen_logits = self.forward_ffhq_D(disc_inp_img, camera)
                losses['G_ffhq_ref_adv'] = torch.nn.functional.softplus(-gen_logits).mean()
        return losses

    def run_G_th1kh_src2tgt_image(self, batch):
        ret = {}
        losses = {}
        with torch.autograd.profiler.record_function('G_th1kh_mv_forward'):
            camera = batch['th1kh_mv_cameras']
            img = batch['th1kh_tgt_imgs']
            gen_img = self.forward_G(batch['th1kh_ref_head_imgs'], camera, cond={
                        'cond_cano': batch['th1kh_cano_secc'], 'cond_src': batch['th1kh_ref_secc'], 'cond_tgt': batch['th1kh_mv_secc'], 
                        'ref_torso_img': batch['th1kh_ref_torso_imgs'], 'bg_img': batch['th1kh_bg_imgs'],
                        'segmap': batch['th1kh_ref_segmaps'],
                        'kp_s':batch['th1kh_ref_kp'], 'kp_d': batch['th1kh_mv_kp'],
                        'target_torso_mask': batch['th1kh_mv_torso_masks_raw'],
                        }, ret=ret)
            if 'losses' in ret: losses.update(ret['losses'])
            
            losses['G_mv_plane_l1_mean'] = (gen_img['plane'][:,:]).detach().abs().mean()
            losses['G_mv_plane_l1_std'] = (gen_img['plane'][:,:]).detach().abs().std()
            if hparams.get("masked_error", True):
                # 之所以用L1不用MSE，原因是mse对mismatch的pixel loss过大，而导致面部细节被忽略，此外还有过模糊的问题
                # 对raw图像，因为deform的原因背景没法全黑，导致这部分mse过高，我们将其mask掉，只计算人脸部分
                losses['G_th1kh_mv_img_mae'] = self.masked_error_loss(gen_img['image'], img, batch['th1kh_mv_torso_masks'], mode='l1', unmasked_weight=0.5)
                losses['G_th1kh_mv_img_lpips'] = self.criterion_lpips(gen_img['image'], img).mean()
            else:
                losses['G_th1kh_mv_img_mae'] = (gen_img['image'] - img).abs().mean()
                losses['G_th1kh_mv_img_lpips'] = self.criterion_lpips(gen_img['image'], img).mean()
            
            # lip loss
            batch_size = len(gen_img['image'])
            lip_mse_loss = 0
            lip_lpips_loss = 0
            for i in range(batch_size):
                xmin, xmax, ymin, ymax = batch['th1kh_mv_lip_rects'][i]
                lip_tgt_imgs = img[i:i+1,:, ymin:ymax,xmin:xmax].contiguous()
                lip_pred_imgs = gen_img['image'][i:i+1,:, ymin:ymax,xmin:xmax].contiguous()
                lip_mse_loss = lip_mse_loss + 1/batch_size * (lip_pred_imgs - lip_tgt_imgs).abs().mean()
                try:
                    lip_lpips_loss = lip_lpips_loss + 1/batch_size * self.criterion_lpips(lip_pred_imgs, lip_tgt_imgs).mean()
                except: pass 
            losses['G_th1kh_mv_img_lip_mae'] = lip_mse_loss
            losses['G_th1kh_mv_img_lip_lpips'] = lip_lpips_loss

            self.gen_tmp_output['th1kh_recon_mv_imgs'] = gen_img['image'].detach()
            self.gen_tmp_output['th1kh_recon_mv_imgs_raw'] = gen_img['image_raw'].detach()
            disc_inp_img = {
                'image': gen_img['image'],
                'image_raw': gen_img['image_raw'],
            }
            gen_logits = self.forward_D(disc_inp_img, camera)
            losses['G_th1kh_mv_adv'] = torch.nn.functional.softplus(-gen_logits).mean()
        return losses

    def forward_D_main(self, batch):
        """
        we update ema this substep.
        """
        FFHQ_DISC_UPDATE_INTERVAL = 4
        losses = {}
        with torch.autograd.profiler.record_function('D_minimize_fake_forward'):
            if 'th1kh_recon_mv_imgs' in self.gen_tmp_output:
                camera = batch['th1kh_mv_cameras']
                disc_inp_img = {
                    'image': self.gen_tmp_output['th1kh_recon_mv_imgs'],
                    'image_raw': self.gen_tmp_output['th1kh_recon_mv_imgs_raw'],
                }
                gen_logits = self.forward_D(disc_inp_img, camera, update_emas=True)
                losses['D_minimize_th1kh_mv_fake'] = torch.nn.functional.softplus(gen_logits).mean()
                if hparams.get("add_ffhq_singe_disc", False) and (self.global_step+1) % FFHQ_DISC_UPDATE_INTERVAL == 0:
                    gen_logits = self.forward_ffhq_D(disc_inp_img, camera, update_emas=True)
                    losses['D_ffhq_minimize_model_pred_mv'] = torch.nn.functional.softplus(gen_logits).mean()

            mv_cameras = batch['th1kh_mv_cameras']
            mv_img_tmp_image = batch['th1kh_tgt_imgs'].detach().requires_grad_(True)
            mv_img_tmp_image_raw = batch['th1kh_mv_head_imgs_raw'].detach().requires_grad_(True)
            th1kh_mv_img_tmp = {'image': mv_img_tmp_image, 'image_raw': mv_img_tmp_image_raw}
            th1kh_mv_logits = self.forward_D(th1kh_mv_img_tmp, mv_cameras)
            losses['D_maximize_th1kh_mv'] = torch.nn.functional.softplus(-th1kh_mv_logits).mean()
            
            if hparams.get("add_ffhq_singe_disc", False) and (self.global_step+1) % FFHQ_DISC_UPDATE_INTERVAL == 0:
                ffhq_ref_img_tmp = {'image': batch['ffhq_com_imgs'].detach().requires_grad_(True),'image_raw': batch['ffhq_com_imgs_raw'].detach().requires_grad_(True)}
                ffhq_ref_logits = self.forward_ffhq_D(ffhq_ref_img_tmp, mv_cameras) # mv_camera will be mul 0 in forward_ffhq_D
                losses['D_ffhq_maximize_gt_ref'] = torch.nn.functional.softplus(-ffhq_ref_logits).mean()

            if (self.global_step+1) % hparams['reg_interval_d'] == 0 and self.training is True:
                with conv2d_gradfix.no_weight_gradients():
                    mv_r1_grads = torch.autograd.grad(outputs=[th1kh_mv_logits.sum()], inputs=[th1kh_mv_img_tmp['image'], th1kh_mv_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                    mv_r1_grads_image = mv_r1_grads[0]
                    mv_r1_grads_image_raw = mv_r1_grads[1]
                    mv_r1_penalty_raw = mv_r1_grads_image_raw.square().sum([1,2,3]).mean()
                    mv_r1_penalty_image = mv_r1_grads_image.square().sum([1,2,3]).mean()
                    losses['D_th1kh_mv_gradient_penalty'] = (mv_r1_penalty_image + mv_r1_penalty_raw) / 2
                if hparams.get("add_ffhq_singe_disc", False) and (self.global_step+1) % FFHQ_DISC_UPDATE_INTERVAL == 0:
                    with conv2d_gradfix.no_weight_gradients():
                        ref_r1_grads = torch.autograd.grad(outputs=[ffhq_ref_logits.sum()], inputs=[ffhq_ref_img_tmp['image'], ffhq_ref_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                        ref_r1_grads_image = ref_r1_grads[0]
                        ref_r1_grads_image_raw = ref_r1_grads[1]
                        ref_r1_penalty_raw = ref_r1_grads_image_raw.square().sum([1,2,3]).mean()
                        ref_r1_penalty_image = ref_r1_grads_image.square().sum([1,2,3]).mean()
                        losses['D_ffhq_gradient_penalty_gt_ref'] = (ref_r1_penalty_image + ref_r1_penalty_raw) / 2

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

        losses = {}
        if optimizer_idx == 0:
            # Train Generator
            self.model.on_train_superresolution()

            losses.update(self.run_G_th1kh_src2src_image(sample))
            losses.update(self.run_G_th1kh_src2tgt_image(sample))
            loss_weights = {
            'G_th1kh_mv_img_mae': hparams['lambda_mse'],
            'G_th1kh_mv_img_lpips': 0.1,
            'G_th1kh_mv_adv': hparams['lambda_th1kh_mv_adv'] if self.global_step >= self.start_adv_iters else 0.,
            'G_th1kh_mv_img_lip_mae': 0.2,
            'G_th1kh_mv_img_lip_lpips': 0.02,

            'G_th1kh_ref_img_mae': hparams['lambda_mse'],
            'G_th1kh_ref_img_lpips': 0.1,
            'G_th1kh_ref_adv': hparams['lambda_th1kh_mv_adv'] if self.global_step >= self.start_adv_iters else 0.,
            'G_th1kh_ref_img_lip_mae': 0.2,
            'G_th1kh_ref_img_lip_lpips': 0.02,

            'facev2v/occlusion_reg_l1': hparams['lam_occlusion_reg_l1'],
            'facev2v/occlusion_2_reg_l1': hparams.get('lam_occlusion_2_reg_l1', 0.),
            'facev2v/occlusion_2_weights_entropy': hparams['lam_occlusion_weights_entropy'],
            }

        elif optimizer_idx == 1:
            # Train Disc
            if self.global_step < hparams["start_adv_iters"] - 10000:
                # start train disc too early is a waste of resource
                return None
            losses.update(self.forward_D_main(sample))
            loss_weights = {
                'D_maximize_ref': 1.0,
                'D_minimize_ref_fake': 1.0,
                'D_ref_gradient_penalty': hparams['lambda_gradient_penalty'] * hparams['reg_interval_d'],
                'D_maximize_mv': 1.0,
                'D_minimize_mv_fake': 1.0,
                'D_mv_gradient_penalty': hparams['lambda_gradient_penalty'] * hparams['reg_interval_d'],

                'D_maximize_th1kh_ref': 1.0,
                'D_minimize_th1kh_ref_fake': 1.0,
                'D_th1kh_ref_gradient_penalty': hparams['lambda_gradient_penalty'] * hparams['reg_interval_d'],
                'D_maximize_th1kh_mv': 1.0,
                'D_minimize_th1kh_mv_fake': 1.0,
                'D_th1kh_mv_gradient_penalty': hparams['lambda_gradient_penalty'] * hparams['reg_interval_d'],
            }
            self.gen_tmp_output = {}
        else:
            return None
        total_loss = sum([loss_weights[k] * v for k, v in losses.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        # total_loss = sum([loss_weights.get(k, 1.0) * v for k, v in losses.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        if len(losses) == 0:
            return None
        return total_loss, losses
    
    #####################
    # Validation
    #####################
    def validation_start(self):
        secc_img2plane_ckpt_dir = hparams.get('reload_head_ckpt', '')
        if secc_img2plane_ckpt_dir != '':
            load_ckpt(self.model.cano_img2plane_backbone, secc_img2plane_ckpt_dir, model_name='model.cano_img2plane_backbone', strict=True)
            load_ckpt(self.model.secc_img2plane_backbone, secc_img2plane_ckpt_dir, model_name='model.secc_img2plane_backbone', strict=True)
            load_ckpt(self.model.decoder, secc_img2plane_ckpt_dir, model_name='model.decoder', strict=True)

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

        losses.update(self.run_G_th1kh_src2tgt_image(sample))
        losses.update(self.run_G_reg(sample))
        losses.update(self.forward_D_main(sample))
        outputs['losses'] = losses
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = tensors_to_scalars(outputs)

        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots'] and rank == 0:

            imgs_ref = sample['th1kh_ref_head_imgs']
            gen_img = self.model.forward(imgs_ref, sample['th1kh_mv_cameras'], cond={'cond_cano': sample['th1kh_cano_secc'],'cond_src': sample['th1kh_ref_secc'], 'cond_tgt': sample['th1kh_mv_secc'],'ref_torso_img': sample['th1kh_ref_torso_imgs'], 'bg_img': sample['th1kh_bg_imgs'],
                        'segmap': sample['th1kh_ref_segmaps'], 'kp_s':sample['th1kh_ref_kp'], 'kp_d': sample['th1kh_mv_kp']}, noise_mode='const')
            gen_img_recon = self.model.forward(imgs_ref, sample['th1kh_ref_cameras'], cond={'cond_cano': sample['th1kh_cano_secc'], 'cond_src': sample['th1kh_ref_secc'], 'cond_tgt': sample['th1kh_ref_secc'],'ref_torso_img': sample['th1kh_ref_torso_imgs'], 'bg_img': sample['th1kh_bg_imgs'],
                        'segmap': sample['th1kh_ref_segmaps'], 'kp_s':sample['th1kh_ref_kp'], 'kp_d': sample['th1kh_ref_kp']}, noise_mode='const')

            imgs_recon = gen_img_recon['image'].permute(0, 2,3,1)
            imgs_recon_raw = filtered_resizing(gen_img_recon['image_raw'], size=512, f=self.resample_filter, filter_mode='antialiased').permute(0, 2,3,1)
            imgs_recon_depth = gen_img_recon['image_depth'].permute(0, 2,3,1)
            imgs_pred_raw = filtered_resizing(gen_img['image_raw'], size=512, f=self.resample_filter, filter_mode='antialiased').permute(0, 2,3,1)
            imgs_pred = gen_img['image'].permute(0, 2,3,1)
            imgs_pred_depth = gen_img['image_depth'].permute(0, 2,3,1)
            imgs_ref = imgs_ref.permute(0,2,3,1)
            imgs_mv = sample['th1kh_tgt_imgs'].permute(0,2,3,1) # [B, H, W, 3]
            imgs_ref_com = sample['th1kh_ref_com_imgs'].permute(0,2,3,1) # [B, H, W, 3]

            for i in range(len(imgs_pred)):
                idx_string = format(i+batch_idx * hparams['batch_size'], "05d")
                base_fn = f"{idx_string}"
                img_ref_mv_recon_pred = torch.cat([imgs_ref_com[i], imgs_mv[i], imgs_recon_raw[i], imgs_pred_raw[i], imgs_recon[i], imgs_pred[i]], dim=1)
                ref_secc = filtered_resizing(sample['th1kh_ref_secc'][i].unsqueeze(0), size=512, f=self.resample_filter, filter_mode='antialiased')[0].permute(1,2,0)
                mv_secc = filtered_resizing(sample['th1kh_mv_secc'][i].unsqueeze(0), size=512, f=self.resample_filter, filter_mode='antialiased')[0].permute(1,2,0)
                img_ref_mv_recon_pred = torch.cat([img_ref_mv_recon_pred, ref_secc, mv_secc], dim=1)
                self.save_rgb_to_fname(img_ref_mv_recon_pred, f"{self.gen_dir}/th1kh_images_rgb_iter{self.global_step}/ref_mv_reconraw_predraw_recon_pred_{base_fn}.png")
                img_depth_recon_pred = torch.cat([imgs_recon_depth[i], imgs_pred_depth[i]], dim=1)
                self.save_depth_to_fname(img_depth_recon_pred, f"{self.gen_dir}/th1kh_images_depth_iter{self.global_step}/recon_pred_{base_fn}.png") 
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)
