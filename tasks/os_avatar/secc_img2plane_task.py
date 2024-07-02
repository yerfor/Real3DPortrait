import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import os
import random 
import copy
import cv2
import math
import lpips

from utils.commons.hparams import hparams
from utils.commons.tensor_utils import tensors_to_scalars, convert_to_np, move_to_cuda
from utils.nn.model_utils import not_requires_grad, num_params
from utils.commons.dataset_utils import data_loader
from utils.nn.schedulers import NoneSchedule
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, restore_weights, restore_opt_state

from tasks.os_avatar.loss_utils.vgg19_loss import VGG19Loss
from tasks.os_avatar.dataset_utils.motion2video_dataset import Motion2Video_Dataset

from tasks.os_avatar.img2plane_task import OSAvatarImg2PlaneTask

from modules.eg3ds.models.triplane import TriPlaneGenerator
from modules.eg3ds.models.dual_discriminator import DualDiscriminator, SingleDiscriminator
from modules.eg3ds.torch_utils.ops import conv2d_gradfix
from modules.eg3ds.torch_utils.ops import upfirdn2d
from modules.eg3ds.models.dual_discriminator import filtered_resizing
from modules.real3d.secc_img2plane import OSAvatarSECC_Img2plane

from deep_3drecon.secc_renderer import SECC_Renderer
from data_util.face3d_helper import Face3DHelper
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.runs.binarizer_nerf import get_lip_rect

from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.edit_secc import blink_eye_for_secc


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
            lr_mul_cano_img2plane = hparams['lr_mul_cano_img2plane'] * min(1.0, num_updates / (hparams['start_adv_iters']+20000))
            self.optimizer[optim_i].param_groups[0]['lr'] = lr_mul_cano_img2plane * self.lr * (hparams.get("lr_decay_rate", 0.95)) ** (num_updates // hparams.get("lr_decay_interval", 5_000)) if num_updates > 6_000 else 0 # cano_img2plane
            self.optimizer[optim_i].param_groups[0]['lr'] = max(5e-6, self.optimizer[optim_i].param_groups[0]['lr'])
            if num_updates >= hparams['stop_update_i2p_iters']:
                self.optimizer[optim_i].param_groups[0]['lr'] = 0.
            self.optimizer[optim_i].param_groups[1]['lr'] = max(5e-6, self.lr * (hparams.get("lr_decay_rate", 0.95)) ** (num_updates // hparams.get("lr_decay_interval", 5_000))) if num_updates > 0 else 0 # secc_img2plane
            self.optimizer[optim_i].param_groups[2]['lr'] = max(5e-6, self.lr * (hparams.get("lr_decay_rate", 0.95)) ** (num_updates // hparams.get("lr_decay_interval", 5_000))) if num_updates > 6_000 else 0 # decoder
            self.optimizer[optim_i].param_groups[3]['lr'] = max(5e-6, self.lr * (hparams.get("lr_decay_rate", 0.95)) ** (num_updates // hparams.get("lr_decay_interval", 5_000))) if num_updates > 30_000 else 0 # sr
        self.optimizer[-1].param_groups[0]['lr'] = max(5e-6, self.lr_d * (hparams.get("lr_decay_rate", 0.95)) ** (num_updates // hparams.get("lr_decay_interval", 5_000))) # for disc
        return self.lr
    

class SECC_Img2PlaneEG3DTask(OSAvatarImg2PlaneTask):
    def __init__(self):
        super().__init__()
        self.seg_model = MediapipeSegmenter()
        self.dataset_cls =  Motion2Video_Dataset
        self.face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='lm68')
        
    def build_model(self):
        self.eg3d_model = TriPlaneGenerator()
        load_ckpt(self.eg3d_model, hparams['pretrained_eg3d_ckpt'], strict=True)
        self.model = OSAvatarSECC_Img2plane()
        self.disc = DualDiscriminator()

        self.cano_img2plane_params = [p for k, p in self.model.cano_img2plane_backbone.named_parameters() if p.requires_grad]
        self.secc_img2plane_params = [p for k, p in self.model.secc_img2plane_backbone.named_parameters() if p.requires_grad]
        self.decoder_params = [p for p in self.model.decoder.parameters() if p.requires_grad]
        self.upsample_params = [p for p in self.model.superresolution.parameters() if p.requires_grad]
        self.disc_params = [p for k, p in self.disc.named_parameters() if p.requires_grad] 

        if hparams.get("add_ffhq_singe_disc", False):
            self.ffhq_disc = DualDiscriminator()
            self.disc_params += [p for k, p in self.ffhq_disc.named_parameters() if p.requires_grad] 
            eg3d_dir = 'checkpoints/geneface2_ckpts/eg3d_baseline_run2'
            load_ckpt(self.ffhq_disc, eg3d_dir, model_name='disc', strict=True)
        self.secc_renderer = SECC_Renderer(512)

        if hparams.get('init_from_ckpt', '') != '': 
            ckpt_dir = hparams['init_from_ckpt']
            try:
                load_ckpt(self.model.cano_img2plane_backbone, ckpt_dir, model_name='model.cano_img2plane_backbone', strict=True)
                load_ckpt(self.model.secc_img2plane_backbone, ckpt_dir, model_name='model.secc_img2plane_backbone', strict=True)
            except:
                # from a img2plane ckpt
                load_ckpt(self.model.cano_img2plane_backbone, ckpt_dir, model_name='model.img2plane_backbone', strict=False)
                # load_ckpt(self.model.cano_img2plane_backbone, ckpt_dir, model_name='model.img2plane_backbone', strict=True)
            load_ckpt(self.model.decoder, ckpt_dir, model_name='model.decoder', strict=True)
            load_ckpt(self.model.superresolution, ckpt_dir, model_name='model.superresolution', strict=False) # false for spade sr
            load_ckpt(self.disc, ckpt_dir, model_name='disc', strict=True)

        return self.model

    def build_optimizer(self, model):
        self.optimizer_gen = optimizer_gen = torch.optim.Adam(
            self.cano_img2plane_params,
            lr=hparams['lr_g'], # we use a 0.5x smaller lr for transformer
            betas=(hparams['optimizer_adam_beta1_g'], hparams['optimizer_adam_beta2_g'])
        )
        self.optimizer_gen.add_param_group({
            'params': self.secc_img2plane_params,
            'lr': hparams['lr_g'],
            'betas': (hparams['optimizer_adam_beta1_g'], hparams['optimizer_adam_beta2_g'])
        })
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
        optimizers = [optimizer_gen, optimizer_disc]
        return optimizers
    
    def build_scheduler(self, optimizer):
        mb_ratio_d = hparams['reg_interval_d'] / (hparams['reg_interval_d']  + 1)
        return ScheduleForLM3DImg2PlaneEG3D(optimizer, hparams['lr_g'], hparams['lr_d'] * mb_ratio_d, hparams['warmup_updates'])
    
    def forward_G(self, img, camera, cond=None, ret=None, update_emas=False, cache_backbone=True, use_cached_backbone=False):
        """
        ref_img: [B, 3, W, H]
        camera: [b, 25], 16 dim c2w, and 9 dim intrinsic
        cond: a dict of cano_secc, tgt_secc, src_secc
        """
        G = self.model
        gen_output = G.forward(img=img, camera=camera, cond=cond, ret=ret, update_emas=update_emas, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone)
        return gen_output

    def forward_D(self, img, camera, update_emas=False):
        D = self.disc
        logits = D.forward(img, camera, update_emas=update_emas)
        return logits
    
    def forward_ffhq_D(self, img, camera, update_emas=False):
        D = self.ffhq_disc
        logits = D.forward(img, 0*camera, update_emas=update_emas)
        return logits
    
    def prepare_batch(self, batch):
        out_batch = {}
        out_batch['th1kh_ref_cameras'] = batch['th1kh_ref_cameras']
        out_batch['th1kh_ref_head_imgs'] = batch['th1kh_ref_head_imgs']
        out_batch['th1kh_ref_head_imgs_raw'] = filtered_resizing(batch['th1kh_ref_head_imgs'], size=hparams['neural_rendering_resolution'], f=self.resample_filter, filter_mode='antialiased')
        out_batch['th1kh_mv_cameras'] = batch['th1kh_mv_cameras']
        out_batch['th1kh_mv_head_imgs'] = batch['th1kh_mv_head_imgs']
        out_batch['th1kh_mv_head_imgs_raw'] = filtered_resizing(batch['th1kh_mv_head_imgs'], size=hparams['neural_rendering_resolution'], f=self.resample_filter, filter_mode='antialiased')
        
        out_batch['th1kh_ref_eulers'] = batch['th1kh_ref_eulers']
        out_batch['th1kh_ref_trans'] = batch['th1kh_ref_trans']
        with torch.no_grad():
            _, out_batch['th1kh_cano_secc'] = self.secc_renderer(batch['th1kh_ref_ids'],batch['th1kh_ref_exps']*0,batch['th1kh_ref_eulers']*0,batch['th1kh_ref_trans']*0)
            _, out_batch['th1kh_ref_secc'] = self.secc_renderer(batch['th1kh_ref_ids'],batch['th1kh_ref_exps'],batch['th1kh_ref_eulers']*0,batch['th1kh_ref_trans']*0)
            _, out_batch['th1kh_mv_secc'] = self.secc_renderer(batch['th1kh_mv_ids'],batch['th1kh_mv_exps'],batch['th1kh_mv_eulers']*0,batch['th1kh_mv_trans']*0)

        if (self.global_step+1) % hparams['reg_interval_g_cond'] == 0:
            if random.random() < hparams.get("pertube_ref_prob", 0.25): # 1/4的可能对ref secc做pertube
                out_batch['th1kh_pertube_secc0'] = out_batch['th1kh_ref_secc'].clone()
                if hparams.get("secc_pertube_mode", 'randn') == 'randn':
                    _, out_batch['th1kh_pertube_secc1'] = self.secc_renderer(batch['th1kh_ref_ids'] + torch.randn_like(batch['th1kh_ref_ids'])*hparams['secc_pertube_randn_scale'],batch['th1kh_ref_exps'] + torch.randn_like(batch['th1kh_ref_exps'])*hparams['secc_pertube_randn_scale'] ,batch['th1kh_ref_eulers']*0,batch['th1kh_ref_trans']*0)
                elif hparams.get("secc_pertube_mode", 'randn') in ['tv', 'laplacian']:
                    _, out_batch['th1kh_pertube_secc1'] = self.secc_renderer(batch['th1kh_ref_ids'],batch['th1kh_ref_pertube_exps_1'],batch['th1kh_ref_eulers']*0,batch['th1kh_ref_trans']*0)
                    _, out_batch['th1kh_pertube_secc2'] = self.secc_renderer(batch['th1kh_ref_ids'],batch['th1kh_ref_pertube_exps_2'],batch['th1kh_ref_eulers']*0,batch['th1kh_ref_trans']*0)
                else: 
                    raise NotImplementedError()
            else:
                out_batch['th1kh_pertube_secc0'] = out_batch['th1kh_mv_secc']
                if hparams.get("secc_pertube_mode", 'randn') == 'randn':
                    _, out_batch['th1kh_pertube_secc1'] = self.secc_renderer(batch['th1kh_mv_ids'] + torch.randn_like(batch['th1kh_mv_ids'])*hparams['secc_pertube_randn_scale'],batch['th1kh_mv_exps'] + torch.randn_like(batch['th1kh_mv_exps'])*hparams['secc_pertube_randn_scale'] ,batch['th1kh_mv_eulers']*0,batch['th1kh_mv_trans']*0)
                elif hparams.get("secc_pertube_mode", 'randn') in ['tv', 'laplacian']:
                    _, out_batch['th1kh_pertube_secc1'] = self.secc_renderer(batch['th1kh_mv_ids'],batch['th1kh_mv_pertube_exps_1'],batch['th1kh_mv_eulers']*0,batch['th1kh_mv_trans']*0)
                    _, out_batch['th1kh_pertube_secc2'] = self.secc_renderer(batch['th1kh_mv_ids'],batch['th1kh_mv_pertube_exps_2'],batch['th1kh_mv_eulers']*0,batch['th1kh_mv_trans']*0)
                else: 
                    raise NotImplementedError()

        if (self.global_step+1) % hparams['reg_interval_g_cond'] == 0:
            blink_secc_lst1 = []
            blink_secc_lst2 = []
            blink_secc_lst3 = []
            for i in range(len(out_batch['th1kh_mv_secc'])):
                if random.random() < 0.25: # 1/4的可能对ref secc做pertube
                    secc = out_batch['th1kh_ref_secc'][i]
                else:
                    secc = out_batch['th1kh_mv_secc'][i]
                blink_percent1 = random.random() * 0.5 # 0~0.5
                blink_percent3 = 0.5 + random.random() * 0.5 # 0.5~1.0
                blink_percent2 = (blink_percent1 + blink_percent3)/2
                try:
                    out_secc1 = blink_eye_for_secc(secc, blink_percent1).to(secc.device)
                    out_secc2 = blink_eye_for_secc(secc, blink_percent2).to(secc.device)
                    out_secc3 = blink_eye_for_secc(secc, blink_percent3).to(secc.device)
                except:
                    print("blink eye for secc failed, use original secc")
                    out_secc1 = copy.deepcopy(secc)
                    out_secc2 = copy.deepcopy(secc)
                    out_secc3 = copy.deepcopy(secc)
                blink_secc_lst1.append(out_secc1)
                blink_secc_lst2.append(out_secc2)
                blink_secc_lst3.append(out_secc3)
            out_batch['th1kh_blink_mv_secc1'] = torch.stack(blink_secc_lst1)
            out_batch['th1kh_blink_mv_secc2'] = torch.stack(blink_secc_lst2)
            out_batch['th1kh_blink_mv_secc3'] = torch.stack(blink_secc_lst3)

        out_batch['th1kh_ref_head_masks'] = batch['th1kh_ref_head_masks'].unsqueeze(1).bool()
        out_batch['th1kh_ref_head_masks_raw'] = torch.nn.functional.interpolate(out_batch['th1kh_ref_head_masks'].float(), size=(128,128), mode='nearest').bool()

        out_batch['th1kh_ref_head_masks_dilate'] = self.dilate_mask(out_batch['th1kh_ref_head_masks'].float(), ksize=41).bool()
        out_batch['th1kh_ref_head_masks_raw_dilate'] = torch.nn.functional.interpolate(out_batch['th1kh_ref_head_masks_dilate'].float(), size=(128,128), mode='nearest').bool()
        
        out_batch['th1kh_mv_head_masks'] = batch['th1kh_mv_head_masks'].unsqueeze(1).bool()
        out_batch['th1kh_mv_head_masks_raw'] = torch.nn.functional.interpolate(out_batch['th1kh_mv_head_masks'].float(), size=(128,128), mode='nearest').bool()

        out_batch['th1kh_mv_head_masks_dilate'] = self.dilate_mask(out_batch['th1kh_mv_head_masks'].float(), ksize=41).long()
        out_batch['th1kh_mv_head_masks_raw_dilate'] = torch.nn.functional.interpolate(out_batch['th1kh_mv_head_masks_dilate'].float(), size=(128,128), mode='nearest').bool()

        WH = 512 # now we only support 512x512
        ref_lm2ds = WH * self.face3d_helper.reconstruct_lm2d(batch['th1kh_ref_ids'],batch['th1kh_ref_exps'],batch['th1kh_ref_eulers'],batch['th1kh_ref_trans']).cpu().numpy()
        mv_lm2ds = WH * self.face3d_helper.reconstruct_lm2d(batch['th1kh_mv_ids'],batch['th1kh_mv_exps'],batch['th1kh_mv_eulers'],batch['th1kh_mv_trans']).cpu().numpy()
        ref_lip_rects = [get_lip_rect(ref_lm2ds[i], WH, WH) for i in range(len(ref_lm2ds))]
        mv_lip_rects = [get_lip_rect(mv_lm2ds[i], WH, WH) for i in range(len(mv_lm2ds))]
        out_batch['th1kh_ref_lip_rects'] = ref_lip_rects
        out_batch['th1kh_mv_lip_rects'] = mv_lip_rects

        return out_batch

    def run_G_th1kh_src2src_image(self, batch):
        """
            不在src2src上训练会导致画质变差、不像说话人, 这很合理, 因为i2p也是这样需要update on ref_mse
            尤其是在靠近src的画质变好, 但同时会导致depth和color在靠近src的时候闪烁.
            解决方法: 算secc2plane pertube loss的时候, 更频繁地在src secc附近计算loss; target到更小的pertube loss
        """

        losses = {}
        ret = {}
        ret['losses'] = {}

        if self.global_step % hparams['update_src2src_interval'] != 0:
            return losses

        with torch.autograd.profiler.record_function('G_th1kh_ref_forward'):
            camera = batch['th1kh_ref_cameras']
            img = batch['th1kh_ref_head_imgs']
            img_raw = batch['th1kh_ref_head_imgs_raw']

            gen_img = self.forward_G(batch['th1kh_ref_head_imgs'], camera, 
                        cond={'cond_cano': batch['th1kh_cano_secc'], 
                            'cond_src': batch['th1kh_ref_secc'], 
                            'cond_tgt': batch['th1kh_ref_secc'], 
                            'ref_head_img': batch['th1kh_ref_head_imgs'], # used for spade sr
                            'ref_alphas': batch['th1kh_ref_head_masks'].float(),
                            'ref_cameras': batch['th1kh_ref_cameras'],
                            },
                        ret=ret)
            losses.update(ret['losses'])
            
            if hparams.get("masked_error", True):
                # 之所以用L1不用MSE, 原因是mse对mismatch的pixel loss过大, 而导致面部细节被忽略, 此外还有过模糊的问题
                # 对mse raw图像, 因为deform的原因背景没法全黑, 导致这部分mse过高, 我们将其mask掉, 只计算人脸部分
                # 在算lpips的时候, 尝试过把非头部mask掉再输入到VGG里面, 但是似乎有点问题, 所以最终没有mask掉非脸
                losses['G_th1kh_ref_img_mae_raw'] = self.masked_error_loss(gen_img['image_raw'], img_raw, batch['th1kh_ref_head_masks_raw_dilate'], mode='l1', unmasked_weight=0.2)
                losses['G_th1kh_ref_img_mae'] = self.masked_error_loss(gen_img['image'], img, batch['th1kh_ref_head_masks_dilate'], mode='l1', unmasked_weight=0.2)
                pred_img_for_vgg = gen_img['image']
                pred_img_raw_for_vgg = gen_img['image_raw']
                losses['G_th1kh_ref_img_lpips'] = self.criterion_lpips(pred_img_for_vgg, img).mean()
                losses['G_th1kh_ref_img_lpips_raw'] = self.criterion_lpips(pred_img_raw_for_vgg, img_raw).mean()
                disc_inp_img = {
                    'image': pred_img_for_vgg,
                    'image_raw': pred_img_raw_for_vgg,
                }
                # lip loss
                batch_size = len(gen_img['image'])
                lip_mse_loss = 0
                lip_lpips_loss = 0
                for i in range(batch_size):
                    xmin, xmax, ymin, ymax = batch['th1kh_ref_lip_rects'][i]
                    lip_tgt_imgs = img[i:i+1,:, ymin:ymax,xmin:xmax].contiguous()
                    lip_pred_imgs = pred_img_for_vgg[i:i+1,:, ymin:ymax,xmin:xmax].contiguous()
                    lip_mse_loss = lip_mse_loss + 1/batch_size * (lip_pred_imgs - lip_tgt_imgs).abs().mean()
                    try:
                        lip_lpips_loss = lip_lpips_loss + 1/batch_size * self.criterion_lpips(lip_pred_imgs, lip_tgt_imgs).mean()
                    except: pass 
                losses['G_th1kh_ref_img_lip_mae'] = lip_mse_loss
                losses['G_th1kh_ref_img_lip_lpips'] = lip_lpips_loss

            else:
                losses['G_th1kh_ref_img_mae_raw'] = (gen_img['image_raw'] - img_raw).abs().mean()
                losses['G_th1kh_ref_img_mae'] = (gen_img['image'] - img).abs().mean()
                losses['G_th1kh_ref_img_lpips'] = self.criterion_lpips(gen_img['image'], img).mean()
                losses['G_th1kh_ref_img_lpips_raw'] = self.criterion_lpips(gen_img['image_raw'], img_raw).mean()
                disc_inp_img = {
                    'image': gen_img['image'],
                    'image_raw': gen_img['image_raw'],
                }

            # ablate后发现, 去掉对ref的weights reg loss, 会导致学到的density比较散, 略微降低画质
            alphas = gen_img['weights_img'].clamp(1e-5, 1 - 1e-5)
            losses['G_th1kh_ref_weights_entropy_loss'] = torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas))
            face_mask = batch['th1kh_ref_head_masks_raw'].bool()
            nonface_mask = ~ batch['th1kh_ref_head_masks_raw'].bool()
            losses['G_th1kh_ref_weights_l1_loss'] = (alphas[nonface_mask]-0).pow(2).mean() + (alphas[face_mask]-1).pow(2).mean()
            
            gen_logits = self.forward_D(disc_inp_img, camera)
            losses['G_th1kh_ref_adv'] = torch.nn.functional.softplus(-gen_logits).mean()
        return losses

    def run_G_th1kh_src2tgt_image(self, batch):
        losses = {}
        ret = {}
        ret['losses'] = {}
        with torch.autograd.profiler.record_function('G_th1kh_mv_forward'):
            camera = batch['th1kh_mv_cameras']
            img = batch['th1kh_mv_head_imgs']
            img_raw = batch['th1kh_mv_head_imgs_raw']

            gen_img = self.forward_G(batch['th1kh_ref_head_imgs'], camera, 
                                cond={'cond_cano': batch['th1kh_cano_secc'], 
                                    'cond_src': batch['th1kh_ref_secc'], 
                                    'cond_tgt': batch['th1kh_mv_secc'], 
                                    'ref_head_img': batch['th1kh_ref_head_imgs'],
                                    'ref_alphas': batch['th1kh_ref_head_masks'].float(),
                                    'ref_cameras': batch['th1kh_ref_cameras'],
                                    }, 
                                ret=ret)
            losses.update(ret['losses'])
            self.gen_tmp_output['th1kh_recon_mv_imgs'] = gen_img['image'].detach()
            self.gen_tmp_output['th1kh_recon_mv_imgs_raw'] = gen_img['image_raw'].detach()
            losses['G_ref_plane_l1_mean'] = (gen_img['plane'][:,:]).detach().abs().mean()
            losses['G_ref_plane_l1_std'] = (gen_img['plane'][:,:]).detach().abs().std()
            
            if hparams.get("masked_error", True):
                # 之所以用L1不用MSE, 原因是mse对mismatch的pixel loss过大, 而导致面部细节被忽略, 此外还有过模糊的问题
                # 对raw图像, 因为deform的原因背景没法全黑, 导致这部分mse过高, 我们将其mask掉, 只计算人脸部分
                losses['G_th1kh_mv_img_mae_raw'] = self.masked_error_loss(gen_img['image_raw'], img_raw, batch['th1kh_mv_head_masks_raw_dilate'], mode='l1', unmasked_weight=0.2)
                losses['G_th1kh_mv_img_mae'] = self.masked_error_loss(gen_img['image'], img, batch['th1kh_mv_head_masks_dilate'], mode='l1', unmasked_weight=0.2)
                pred_img_for_vgg = gen_img['image']
                pred_img_raw_for_vgg = gen_img['image_raw']
                losses['G_th1kh_mv_img_lpips'] = self.criterion_lpips(pred_img_for_vgg, img).mean()
                losses['G_th1kh_mv_img_lpips_raw'] = self.criterion_lpips(pred_img_raw_for_vgg, img_raw).mean()

                # emphasize lip loss
                batch_size = len(gen_img['image'])
                lip_mse_loss = 0
                lip_lpips_loss = 0
                for i in range(batch_size):
                    xmin, xmax, ymin, ymax = batch['th1kh_mv_lip_rects'][i]
                    lip_tgt_imgs = img[i:i+1,:, ymin:ymax,xmin:xmax].contiguous()
                    lip_pred_imgs = pred_img_for_vgg[i:i+1,:, ymin:ymax,xmin:xmax].contiguous()
                    lip_mse_loss = lip_mse_loss + 1/batch_size * (lip_pred_imgs - lip_tgt_imgs).abs().mean()
                    try:
                        lip_lpips_loss = lip_lpips_loss + 1/batch_size * self.criterion_lpips(lip_pred_imgs, lip_tgt_imgs).mean()
                    except: pass 
                losses['G_th1kh_mv_img_lip_mae'] = lip_mse_loss
                losses['G_th1kh_mv_img_lip_lpips'] = lip_lpips_loss

                self.gen_tmp_output['th1kh_recon_mv_imgs'] = pred_img_for_vgg.detach()
                self.gen_tmp_output['th1kh_recon_mv_imgs_raw'] = pred_img_raw_for_vgg.detach()
                disc_inp_img = {
                    'image': pred_img_for_vgg,
                    'image_raw': pred_img_raw_for_vgg,
                }

            else:
                losses['G_th1kh_mv_img_mae_raw'] = (gen_img['image_raw'] - img_raw).abs().mean()
                losses['G_th1kh_mv_img_mae'] = (gen_img['image'] - img).abs().mean()
                losses['G_th1kh_mv_img_lpips'] = self.criterion_lpips(gen_img['image'], img).mean()
                losses['G_th1kh_mv_img_lpips_raw'] = self.criterion_lpips(gen_img['image_raw'], img_raw).mean()
                disc_inp_img = {
                    'image': gen_img['image'],
                    'image_raw': gen_img['image_raw'],
                }

            alphas = gen_img['weights_img'].clamp(1e-5, 1 - 1e-5)
            losses['G_th1kh_mv_weights_entropy_loss'] = torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas))
            face_mask = batch['th1kh_mv_head_masks_raw'].bool()
            nonface_mask = ~ batch['th1kh_mv_head_masks_raw'].bool()
            losses['G_th1kh_mv_weights_l1_loss'] = (alphas[nonface_mask]-0).pow(2).mean() + (alphas[face_mask]-1).pow(2).mean()
            
            gen_logits = self.forward_D(disc_inp_img, camera)
            losses['G_th1kh_mv_adv'] = torch.nn.functional.softplus(-gen_logits).mean()
            if hparams.get("add_ffhq_singe_disc", False):
                gen_logits = self.forward_ffhq_D(disc_inp_img, camera)
                losses['G_ffhq_adv_maxmimize_model_pred_mv'] = torch.nn.functional.softplus(-gen_logits).mean()
        return losses

    def run_G_reg(self, batch):
        losses = {}
        imgs = batch['th1kh_ref_head_imgs']
        if (self.global_step+1) % hparams['reg_interval_g'] == 0:
            with torch.autograd.profiler.record_function('G_regularize_forward'):
                cond={'cond_cano': batch['th1kh_cano_secc'],
                    'cond_src': batch['th1kh_ref_secc'], 
                    'cond_tgt': batch['th1kh_mv_secc'],
                    'ref_cameras': batch['th1kh_ref_cameras'],
                    'ref_alphas': batch['th1kh_ref_head_masks'].float(),
                    }
                initial_coordinates = torch.rand((imgs.shape[0], 1000, 3), device=imgs.device) - 0.5 # [-0.5,0.5]
                perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * 5e-3
                all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
                source_sigma = self.model.sample(coordinates=all_coordinates, directions=torch.randn_like(all_coordinates), img=imgs, cond=cond, update_emas=False)['sigma']
                source_sigma_initial = source_sigma[:, :source_sigma.shape[1]//2]
                source_sigma_perturbed = source_sigma[:, source_sigma.shape[1]//2:]
                density_reg_loss = torch.nn.functional.l1_loss(source_sigma_initial, source_sigma_perturbed)

                # we want the pertubed position has similar density
                losses['G_th1kh_regularize_density_l1'] = density_reg_loss

        return losses
    
    def run_G_reg_cond(self, batch):
        losses = {}
        if (self.global_step+1) % hparams['reg_interval_g_cond'] == 0:
            # Reg pertube ref/mv_secc, see prepare_batch, we have 25% prob pertube ref and 75% pertube mv.
            cond = {'cond_cano': batch['th1kh_cano_secc'], 'cond_src': batch['th1kh_ref_secc'], 'cond_tgt': batch['th1kh_pertube_secc0']}
            pertube_cond = {'cond_cano': batch['th1kh_cano_secc'], 'cond_src': batch['th1kh_ref_secc'], 'cond_tgt': batch['th1kh_pertube_secc1']}
            secc_plane = self.model.cal_secc_plane(cond)
            pertube_secc_plane = self.model.cal_secc_plane(pertube_cond)
            with torch.autograd.profiler.record_function('G_regularize_forward'):
                if hparams.get("secc_pertube_mode", 'randn') in ['randn', 'tv']:
                    secc_reg_loss = torch.nn.functional.l1_loss(secc_plane, pertube_secc_plane)
                    # we want the pertubed position has similar density
                elif hparams.get("secc_pertube_mode", 'randn') == 'laplacian':
                    pertube_cond2 = { 'cond_cano': batch['th1kh_cano_secc'], 'cond_src': batch['th1kh_ref_secc'], 'cond_tgt': batch['th1kh_pertube_secc2']}
                    pertube_secc_plane2 = self.model.cal_secc_plane(pertube_cond2)
                    interpolate_secc_plane = (pertube_secc_plane + pertube_secc_plane2) / 2
                    secc_reg_loss = torch.nn.functional.l1_loss(secc_plane, interpolate_secc_plane)
                else: raise NotImplementedError()
                losses['G_th1kh_regularize_pertube_secc_mae'] = secc_reg_loss
            
            # Reg blinks
            blink_cond1 = {'cond_cano': batch['th1kh_cano_secc'], 'cond_src': batch['th1kh_ref_secc'], 'cond_tgt': batch['th1kh_blink_mv_secc1']}
            blink_cond2 = {'cond_cano': batch['th1kh_cano_secc'], 'cond_src': batch['th1kh_ref_secc'], 'cond_tgt': batch['th1kh_blink_mv_secc2']}
            blink_cond3 = {'cond_cano': batch['th1kh_cano_secc'], 'cond_src': batch['th1kh_ref_secc'], 'cond_tgt': batch['th1kh_blink_mv_secc3']}
            blink_secc_plane1 = self.model.cal_secc_plane(blink_cond1)
            blink_secc_plane2 = self.model.cal_secc_plane(blink_cond2)
            blink_secc_plane3 = self.model.cal_secc_plane(blink_cond3)
            interpolate_blink_secc_plane = (blink_secc_plane1 + blink_secc_plane3) / 2
            blink_reg_loss = torch.nn.functional.l1_loss(blink_secc_plane2, interpolate_blink_secc_plane)
            losses['G_th1kh_regularize_blink_secc_mae'] = blink_reg_loss

        return losses
    
    def forward_D_main(self, batch):
        """
        we update ema this substep.
        """
        FFHQ_DISC_UPDATE_INTERVAL = 4
        losses = {}
        with torch.autograd.profiler.record_function('D_minimize_fake_forward'):
            # gt ref img & Dmain
            ref_cameras = batch['th1kh_ref_cameras']
            ref_img_tmp_image = batch['th1kh_ref_head_imgs'].detach().requires_grad_(True)
            ref_img_tmp_image_raw = batch['th1kh_ref_head_imgs_raw'].detach().requires_grad_(True)
            th1kh_ref_img_tmp = {'image': ref_img_tmp_image, 'image_raw': ref_img_tmp_image_raw}
            th1kh_ref_logits = self.forward_D(th1kh_ref_img_tmp, ref_cameras)
            losses['D_th1kh_maximize_gt_ref'] = torch.nn.functional.softplus(-th1kh_ref_logits).mean()
            if hparams.get("add_ffhq_singe_disc", False) and (self.global_step+1) % FFHQ_DISC_UPDATE_INTERVAL == 0:
                ffhq_ref_img_tmp = {'image': batch['ffhq_head_imgs'].detach().requires_grad_(True),'image_raw': batch['ffhq_head_imgs_raw'].detach().requires_grad_(True)}
                ffhq_ref_logits = self.forward_ffhq_D(ffhq_ref_img_tmp, ref_cameras) # ref_camera will be mul 0 in forward_ffhq_D
                losses['D_ffhq_maximize_gt_ref'] = torch.nn.functional.softplus(-ffhq_ref_logits).mean()

            # gt ref img & gradient penalty
            if (self.global_step + 1) % hparams['reg_interval_d'] == 0 and self.training is True:
                with conv2d_gradfix.no_weight_gradients():
                    ref_r1_grads = torch.autograd.grad(outputs=[th1kh_ref_logits.sum()], inputs=[th1kh_ref_img_tmp['image'], th1kh_ref_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                    ref_r1_grads_image = ref_r1_grads[0]
                    ref_r1_grads_image_raw = ref_r1_grads[1]
                    ref_r1_penalty_raw = ref_r1_grads_image_raw.square().sum([1,2,3]).mean()
                    ref_r1_penalty_image = ref_r1_grads_image.square().sum([1,2,3]).mean()
                    losses['D_th1kh_gradient_penalty_gt_ref'] = (ref_r1_penalty_image + ref_r1_penalty_raw) / 2
                if hparams.get("add_ffhq_singe_disc", False):
                    with conv2d_gradfix.no_weight_gradients():
                        ref_r1_grads = torch.autograd.grad(outputs=[ffhq_ref_logits.sum()], inputs=[ffhq_ref_img_tmp['image'], ffhq_ref_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                        ref_r1_grads_image = ref_r1_grads[0]
                        ref_r1_grads_image_raw = ref_r1_grads[1]
                        ref_r1_penalty_raw = ref_r1_grads_image_raw.square().sum([1,2,3]).mean()
                        ref_r1_penalty_image = ref_r1_grads_image.square().sum([1,2,3]).mean()
                        losses['D_ffhq_gradient_penalty_gt_ref'] = (ref_r1_penalty_image + ref_r1_penalty_raw) / 2

            # pred mv img & D minimize
            if 'th1kh_recon_mv_imgs' in self.gen_tmp_output:
                camera = batch['th1kh_mv_cameras']
                disc_inp_img = {
                    'image': self.gen_tmp_output['th1kh_recon_mv_imgs'],
                    'image_raw': self.gen_tmp_output['th1kh_recon_mv_imgs_raw'],
                }
                gen_logits = self.forward_D(disc_inp_img, camera, update_emas=True)
                losses['D_th1kh_minimize_model_pred_mv'] = torch.nn.functional.softplus(gen_logits).mean()
                if hparams.get("add_ffhq_singe_disc", False) and (self.global_step+1) % FFHQ_DISC_UPDATE_INTERVAL == 0:
                    gen_logits = self.forward_ffhq_D(disc_inp_img, camera, update_emas=True)
                    losses['D_ffhq_minimize_model_pred_mv'] = torch.nn.functional.softplus(gen_logits).mean()

            # gt mv img & D maximize
            mv_cameras = batch['th1kh_mv_cameras']
            mv_img_tmp_image = batch['th1kh_mv_head_imgs'].detach().requires_grad_(True)
            mv_img_tmp_image_raw = batch['th1kh_mv_head_imgs_raw'].detach().requires_grad_(True)
            th1kh_mv_img_tmp = {'image': mv_img_tmp_image, 'image_raw': mv_img_tmp_image_raw}
            th1kh_mv_logits = self.forward_D(th1kh_mv_img_tmp, mv_cameras)
            losses['D_th1kh_maximize_gt_mv'] = torch.nn.functional.softplus(-th1kh_mv_logits).mean()
            
            # gt mv img & gradient penalty
            if (self.global_step + 1) % hparams['reg_interval_d'] == 0 and self.training is True:
                with conv2d_gradfix.no_weight_gradients():
                    mv_r1_grads = torch.autograd.grad(outputs=[th1kh_mv_logits.sum()], inputs=[th1kh_mv_img_tmp['image'], th1kh_mv_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                    mv_r1_grads_image = mv_r1_grads[0]
                    mv_r1_grads_image_raw = mv_r1_grads[1]
                    mv_r1_penalty_raw = mv_r1_grads_image_raw.square().sum([1,2,3]).mean()
                    mv_r1_penalty_image = mv_r1_grads_image.square().sum([1,2,3]).mean()
                    losses['D_th1kh_gradient_penalty_gt_mv'] = (mv_r1_penalty_image + mv_r1_penalty_raw) / 2

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
            if hparams['two_stage_training']:
                if self.global_step >= self.start_adv_iters:
                    # only the resolution module requires grad
                    self.model.on_train_superresolution()
                    if hparams.get('also_update_decoder'):
                        self.model.decoder.requires_grad_(True)
                else:
                    # only the nerf module requires grad
                    self.model.on_train_full_model()
            else:
                self.model.on_train_full_model()

            losses.update(self.run_G_th1kh_src2src_image(sample)) # 提升identity similarity, 很必要, 否则会相似度变差
            losses.update(self.run_G_th1kh_src2tgt_image(sample))
            losses.update(self.run_G_reg(sample))
            losses.update(self.run_G_reg_cond(sample))
            loss_weights = {
            'G_th1kh_ref_img_mae': hparams.get("lambda_mse", 1.0),
            'G_th1kh_ref_img_mae_raw': hparams.get("lambda_mse", 1.0),
            'G_th1kh_ref_img_lpips': 0.1,
            'G_th1kh_ref_img_lpips_raw': 0.1,
            'G_th1kh_ref_adv': hparams['lambda_th1kh_mv_adv'] if self.global_step >= self.start_adv_iters else 0.,
            'G_th1kh_ref_weights_l1_loss': hparams.get("lambda_weights_l1", 0.5),
            'G_th1kh_ref_weights_entropy_loss': hparams.get("lambda_weights_entropy", 0.05),
            
            'G_th1kh_mv_img_mae': hparams.get("lambda_mse", 1.0),
            'G_th1kh_mv_img_mae_raw': hparams.get("lambda_mse", 1.0),
            'G_th1kh_mv_img_lpips': 0.1,
            'G_th1kh_mv_img_lpips_raw': 0.1,
            'G_th1kh_mv_adv': hparams['lambda_th1kh_mv_adv'] if self.global_step >= self.start_adv_iters else 0.,
            'G_th1kh_mv_weights_l1_loss': hparams.get("lambda_weights_l1", 0.3),
            'G_th1kh_mv_weights_entropy_loss': hparams.get("lambda_weights_entropy", 0.01),

            'G_th1kh_ref_img_lip_mae': 0.5,
            'G_th1kh_ref_img_lip_lpips': 0.05,
            'G_th1kh_mv_img_lip_mae': 0.5,
            'G_th1kh_mv_img_lip_lpips': 0.05,

            'G_th1kh_regularize_density_l1': hparams['lambda_density_reg'] * hparams['reg_interval_g'],
            'G_ffhq_adv_maxmimize_model_pred_mv': hparams['lambda_ffhq_mv_adv'] if self.global_step >= self.start_adv_iters else 0.,
            'secc_deform_l1_losses': 0.1,
            }
            
            if 'G_th1kh_regularize_pertube_secc_mae' in losses:
                target_pertube_blink_secc_loss = hparams.get('target_pertube_blink_secc_loss', 0.15)
                target_pertube_secc_loss = hparams.get('target_pertube_secc_loss', 0.15)
                current_pertube_blink_secc_loss = losses['G_th1kh_regularize_blink_secc_mae'].item()
                current_pertube_secc_loss = losses['G_th1kh_regularize_pertube_secc_mae'].item()
                grad_lambda_pertube_blink_secc = (math.log10(current_pertube_blink_secc_loss+1e-15) - math.log10(target_pertube_blink_secc_loss+1e-15)) # 如果需要增大lambda_pertube_secc,  则current_loss大于targt, grad值大于0
                grad_lambda_pertube_secc = (math.log10(current_pertube_secc_loss+1e-15) - math.log10(target_pertube_secc_loss+1e-15)) # 如果需要增大lambda_pertube_secc,  则current_loss大于targt, grad值大于0
                lr_lambda_pertube_secc = hparams.get('lr_lambda_pertube_secc', 0.01)
                self.model.lambda_pertube_blink_secc.data = self.model.lambda_pertube_blink_secc.data + grad_lambda_pertube_blink_secc * lr_lambda_pertube_secc
                self.model.lambda_pertube_blink_secc.data.clamp_(0, 2.)
                self.model.lambda_pertube_secc.data = self.model.lambda_pertube_secc.data + grad_lambda_pertube_secc * lr_lambda_pertube_secc
                self.model.lambda_pertube_secc.data.clamp_(0, 0.2)

                if hparams['target_pertube_secc_loss'] == 0.:
                    self.model.lambda_pertube_secc.data = self.model.lambda_pertube_secc.data * 0.
                if hparams['target_pertube_blink_secc_loss'] == 0.:
                    self.model.lambda_pertube_blink_secc.data = self.model.lambda_pertube_blink_secc.data * 0.

                losses['lambda_pertube_blink_secc'] = self.model.lambda_pertube_blink_secc.item()
                losses['lambda_pertube_secc'] = self.model.lambda_pertube_secc.item()
                loss_weights['G_th1kh_regularize_pertube_secc_mae'] = self.model.lambda_pertube_secc.item() * hparams['reg_interval_g_cond'] # 把新的lambda更新到loss_weights里面
                loss_weights['G_th1kh_regularize_blink_secc_mae'] = self.model.lambda_pertube_blink_secc.item() * hparams['reg_interval_g_cond'] # 把新的lambda更新到loss_weights里面

            if hparams.get("disable_highreso_at_stage1", False) and hparams['two_stage_training'] and self.global_step >= self.start_adv_iters:
                loss_weights['G_th1kh_mv_img_mae'] = 0.
                loss_weights['G_th1kh_mv_img_lpips'] = 0.

        elif optimizer_idx == 1:
            # Train Disc
            if self.global_step < hparams["start_adv_iters"] - 10000:
                # start train disc too early is a waste of resource
                return None
            losses.update(self.forward_D_main(sample))
            loss_weights = {
                'D_th1kh_maximize_gt_ref': 1.0,
                'D_ffhq_maximize_gt_ref': 1.0,
                'D_th1kh_maximize_gt_mv': 1.0,
                'D_th1kh_minimize_model_pred_mv': 1.0,
                'D_ffhq_minimize_model_pred_mv': 1.0,

                'D_th1kh_gradient_penalty_gt_ref': hparams['lambda_gradient_penalty'] * hparams['reg_interval_d'],
                'D_th1kh_gradient_penalty_gt_mv': hparams['lambda_gradient_penalty'] * hparams['reg_interval_d'],
                'D_ffhq_gradient_penalty_gt_ref': hparams['lambda_gradient_penalty'] * hparams['reg_interval_d'],
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
        self.gen_dir = os.path.join(hparams['work_dir'], f'validation_results')
        os.makedirs(self.gen_dir, exist_ok=True)

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        self.gen_dir = os.path.join(hparams['work_dir'], f'validation_results')
        os.makedirs(self.gen_dir, exist_ok=True)

        outputs = {}
        losses = {}
        if len(sample) == 0:
            return None
        sample = self.prepare_batch(sample)
        rank = 0 if len(set(os.environ['CUDA_VISIBLE_DEVICES'].split(","))) == 1 else dist.get_rank()

        losses.update(self.run_G_th1kh_src2tgt_image(sample))
        losses.update(self.run_G_reg(sample))
        losses.update(self.run_G_reg_cond(sample))
        losses.update(self.forward_D_main(sample))
        outputs['losses'] = losses
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = tensors_to_scalars(outputs)

        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots'] and rank == 0:

            imgs_ref = sample['th1kh_ref_head_imgs']
            gen_img = self.model.forward(imgs_ref, sample['th1kh_mv_cameras'], 
                                        cond={'cond_cano': sample['th1kh_cano_secc'],
                                            'cond_src': sample['th1kh_ref_secc'], 
                                            'cond_tgt': sample['th1kh_mv_secc'], 
                                            'ref_head_img': imgs_ref,
                                            'ref_cameras': sample['th1kh_ref_cameras'],
                                            'ref_alphas': sample['th1kh_ref_head_masks'].float(),
                                            }, noise_mode='const')
            gen_img_recon = self.model.forward(imgs_ref, sample['th1kh_ref_cameras'], 
                                        cond={'cond_cano': sample['th1kh_cano_secc'], 
                                            'cond_src': sample['th1kh_ref_secc'], 
                                            'cond_tgt': sample['th1kh_ref_secc'], 
                                            'ref_head_img': imgs_ref,
                                            'ref_cameras': sample['th1kh_ref_cameras'],
                                            'ref_alphas': sample['th1kh_ref_head_masks'].float(),
                                            }, noise_mode='const')

            imgs_recon = gen_img_recon['image'].permute(0, 2,3,1)
            imgs_recon_raw = filtered_resizing(gen_img_recon['image_raw'], size=512, f=self.resample_filter, filter_mode='antialiased').permute(0, 2,3,1)
            imgs_recon_depth = gen_img_recon['image_depth'].permute(0, 2,3,1)
            imgs_pred_raw = filtered_resizing(gen_img['image_raw'], size=512, f=self.resample_filter, filter_mode='antialiased').permute(0, 2,3,1)
            imgs_pred = gen_img['image'].permute(0, 2,3,1)
            imgs_pred_depth = gen_img['image_depth'].permute(0, 2,3,1)
            imgs_ref = imgs_ref.permute(0,2,3,1)
            imgs_mv = sample['th1kh_mv_head_imgs'].permute(0,2,3,1) # [B, H, W, 3]

            for i in range(len(imgs_pred)):
                idx_string = format(i+batch_idx * hparams['batch_size'], "05d")
                base_fn = f"{idx_string}"
                img_ref_mv_recon_pred = torch.cat([imgs_ref[i], imgs_mv[i], imgs_recon_raw[i], imgs_pred_raw[i], imgs_recon[i], imgs_pred[i]], dim=1)
                ref_secc = filtered_resizing(sample['th1kh_ref_secc'][i].unsqueeze(0), size=512, f=self.resample_filter, filter_mode='antialiased')[0].permute(1,2,0)
                mv_secc = filtered_resizing(sample['th1kh_mv_secc'][i].unsqueeze(0), size=512, f=self.resample_filter, filter_mode='antialiased')[0].permute(1,2,0)
                img_ref_mv_recon_pred = torch.cat([img_ref_mv_recon_pred, ref_secc], dim=1)
                img_ref_mv_recon_pred = torch.cat([img_ref_mv_recon_pred, mv_secc], dim=1)
                self.save_rgb_to_fname(img_ref_mv_recon_pred, f"{self.gen_dir}/th1kh_images_rgb_iter{self.global_step}/ref_mv_reconraw_predraw_recon_pred_{base_fn}.png")
                img_depth_recon_pred = torch.cat([imgs_recon_depth[i], imgs_pred_depth[i]], dim=1)
                self.save_depth_to_fname(img_depth_recon_pred, f"{self.gen_dir}/th1kh_images_depth_iter{self.global_step}/recon_pred_{base_fn}.png") 
        
        if batch_idx == 0 and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            image_name = "data/raw/examples/Macron.png"
            imgs_ref = cv2.imread(image_name)
            img = load_img_to_512_hwc_array(image_name)
            segmap = self.seg_model._cal_seg_map(img)
            head_img = self.seg_model._seg_out_img_with_segmap(img, segmap, mode='head')[0]
            head_mask = segmap[[1,3,5] , :, :].sum(axis=0)[None,:] > 0.5 # glasses 也属于others
            head_mask = torch.tensor(head_mask).float().reshape([1,1,512,512]).cuda()
            imgs_ref = ((torch.tensor(head_img) - 127.5)/127.5).float().unsqueeze(0).permute(0, 3, 1,2).cuda() # [b,c,h,w]

            from data_gen.utils.process_image.fit_3dmm_landmark import fit_3dmm_for_a_image
            coeff_dict = fit_3dmm_for_a_image(image_name, save=False)
            id = torch.tensor(coeff_dict['id']).float().cuda().reshape([1, 80])
            exp = torch.tensor(coeff_dict['exp']).float().cuda().reshape([1, 64])
            with torch.no_grad():
                _, cano_secc = self.secc_renderer(id,exp*0,sample['th1kh_ref_eulers']*0,sample['th1kh_ref_trans']*0)
                _, ref_secc = self.secc_renderer(id,exp,sample['th1kh_ref_eulers']*0,sample['th1kh_ref_trans']*0)

            gen_img = self.model.forward(imgs_ref, sample['th1kh_mv_cameras'][0:1], 
                        cond={'cond_cano': cano_secc,
                        'cond_src': ref_secc, 
                        'cond_tgt': ref_secc, 
                        'ref_head_img': imgs_ref,
                        'ref_cameras': sample['th1kh_mv_cameras'][0:1],
                        'ref_alphas': head_mask}, 
                        noise_mode='const')
            img = gen_img['image'].permute(0, 2,3,1)[0]
            self.save_rgb_to_fname(img, f"{self.gen_dir}/ood_images_rgb_iter{self.global_step}/May.png")

        return outputs
    
    def masked_error_loss(self, img_pred, img_gt, mask, unmasked_weight=0.1, mode='l1'):
        # mask: [B, 1, H, W]
        # 对raw图像, 因为deform的原因背景没法全黑, 导致这部分mse过高, 我们将其mask掉, 只计算人脸部分
        masked_weight = 1.0
        weight_mask = mask.float() * masked_weight + (~mask).float() * unmasked_weight
        if mode == 'l1':
            error = (img_pred - img_gt).abs().sum(dim=1, keepdim=True) * weight_mask
        else:
            error = (img_pred - img_gt).pow(2).sum(dim=1, keepdim=True) * weight_mask
        error.clamp_(0, max(0.5, error.quantile(0.8).item())) # clamp掉较高loss的pixel, 避免姿态没对齐的pixel导致的异常值占主导影响训练
        loss = error.mean()
        return loss

    def set_unmasked_to_black(self, img, mask):
        out_img = img * mask.float() - (~mask).float() # -1 denotes black
        return out_img

    def dilate(self, bin_img, ksize=5, mode='max_pool'):
        """
        mode: max_pool or avg_pool
        """
        # bin_img, [b, 1, h, w]
        pad = (ksize-1)//2
        bin_img = F.pad(bin_img, pad=[pad,pad,pad,pad], mode='reflect')
        if mode == 'max_pool':
            out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
        else:
            out = F.avg_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
        return out
        
    def dilate_mask(self, mask, ksize=21):
        mask = self.dilate(mask, ksize=ksize, mode='max_pool')
        return mask

    def validation_end(self, outputs):
        return super().validation_end(outputs)
