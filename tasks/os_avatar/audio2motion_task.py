import torch
import torch.nn.functional as F
import numpy as np
import os
from einops import rearrange
import random

from utils.commons.base_task import BaseTask
from utils.commons.dataset_utils import data_loader
from utils.commons.hparams import hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.tensor_utils import tensors_to_scalars, convert_to_np
from utils.nn.model_utils import print_arch, get_device_of_model, not_requires_grad
from utils.nn.schedulers import ExponentialSchedule
from utils.nn.grad import get_grad_norm
from utils.nn.model_utils import print_arch, num_params
from utils.commons.face_alignment_utils import mouth_idx_in_mediapipe_mesh

from modules.audio2motion.vae import VAEModel, PitchContourVAEModel
from tasks.os_avatar.dataset_utils.audio2motion_dataset import Audio2Motion_Dataset
from data_util.face3d_helper import Face3DHelper
from data_gen.utils.mp_feature_extractors.face_landmarker import index_lm68_from_lm478
from modules.syncnet.models import LandmarkHubertSyncNet


class Audio2MotionTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = Audio2Motion_Dataset
        if hparams["motion_type"] == 'id_exp':
            self.in_out_dim = 80 + 64
        elif hparams["motion_type"] == 'exp':
            self.in_out_dim = 64

    def build_model(self):
        if hparams['audio_type'] == 'hubert':
            audio_in_dim = 1024 # hubert
        elif hparams['audio_type'] == 'mfcc':
            audio_in_dim = 13 # hubert

        if hparams.get("use_pitch", False) is True:
            self.model = PitchContourVAEModel(hparams, in_out_dim=self.in_out_dim, audio_in_dim=audio_in_dim, use_prior_flow=hparams.get("use_flow", True))
        else:
            self.model = VAEModel(in_out_dim=self.in_out_dim, audio_in_dim=audio_in_dim, use_prior_flow=hparams.get("use_flow", True))
        
        if hparams.get('init_from_ckpt', '') != '': 
            ckpt_dir = hparams.get('init_from_ckpt', '')
            load_ckpt(self.model, ckpt_dir, model_name='model', strict=True)
            
        self.face3d_helper = Face3DHelper(keypoint_mode='mediapipe', use_gpu=False)
        lm_dim = 468*3 # lip part in idexp_lm3d
        # lm_dim = 20*3 # lip part in idexp_lm3d
        hparams['syncnet_num_layers_per_block'] = 3
        hparams['syncnet_base_hid_size'] = 128
        hparams['syncnet_out_hid_size'] = 1024
        self.syncnet = LandmarkHubertSyncNet(lm_dim, audio_in_dim, num_layers_per_block=hparams['syncnet_num_layers_per_block'], base_hid_size=hparams['syncnet_base_hid_size'], out_dim=hparams['syncnet_out_hid_size'])
        if hparams['syncnet_ckpt_dir']:
            load_ckpt(self.syncnet, hparams['syncnet_ckpt_dir'])
        return self.model
    
    def on_train_start(self):
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        for n, m in self.model.vae.named_children():
            num_params(m, model_name='vae.'+n)

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']))
        return optimizer

    def build_scheduler(self, optimizer):
        return ExponentialSchedule(optimizer, hparams['lr'], hparams['warmup_updates'])

    @data_loader
    def train_dataloader(self):
        if hparams['ds_name'] == 'Concat_voxceleb2_CMLR':
            train_dataset1 = self.dataset_cls(prefix='train', data_dir='data/binary/voxceleb2_audio2motion_kv')
            train_dataset2 = self.dataset_cls(prefix='train', data_dir='data/binary/CMLR_audio2motion_kv')
            train_dataset = BaseConcatDataset([train_dataset1,train_dataset2], prefix='train')
        elif hparams['ds_name'] == 'Weighted_Concat_voxceleb2_CMLR':
            train_dataset1 = self.dataset_cls(prefix='train', data_dir='data/binary/voxceleb2_audio2motion_kv')
            train_dataset2 = self.dataset_cls(prefix='train', data_dir='data/binary/CMLR_audio2motion_kv')
            train_dataset = WeightedConcatDataset([train_dataset1,train_dataset2], [0.5, 0.5], prefix='train')
        else:
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

    ##########################
    # training and validation
    ##########################
    def run_model(self, sample, infer=False, temperature=1.0, sync_batch_size=1024):
        """
        render or train on a single-frame
        :param sample: a batch of data
        :param infer: bool, run in infer mode
        :return:
            if not infer:
                return losses, model_out
            if infer:
                return model_out
        """
        model_out = {}
        if hparams['audio_type'] == 'hubert':
            sample['audio'] = sample['hubert']
        elif hparams['audio_type'] == 'mfcc':
            sample['audio'] = sample['mfcc'] / 100
        elif hparams['audio_type'] == 'mel':
            sample['audio'] = sample['mel'] # [b, 2*t, 1024]

        if hparams.get("blink_mode", 'none') != 'none': # eye_area_percnet or blink_unit
            blink = F.interpolate(sample[hparams['blink_mode']].permute(0,2,1).float(), scale_factor=2).permute(0,2,1).long()
            sample['blink'] = blink
        bs = sample['audio'].shape[0]

        if infer:
            self.model(sample, model_out, train=False, temperature=temperature)
            return model_out

        else:
            losses_out = {}
            if hparams["motion_type"] == 'id_exp':
                x_gt = torch.cat([sample['id'], sample['exp']],dim=-1)
                sample['y'] = x_gt
                self.model(sample, model_out, train=True)
                x_pred = model_out['pred'].reshape([bs, -1, 80+64])
                x_mask = model_out['mask'].reshape([bs, -1])
                losses_out['kl'] = model_out['loss_kl']
                id_pred = x_pred[:, :, :80]
                exp_pred = x_pred[:, :, 80:]
                losses_out['lap_id'] = self.lap_loss(id_pred, x_mask)
                losses_out['lap_exp'] = self.lap_loss(exp_pred, x_mask)
                pred_idexp_lm3d = self.face3d_helper.reconstruct_idexp_lm3d(id_pred, exp_pred).reshape([bs, x_mask.shape[1], -1])
                gt_idexp_lm3d = self.face3d_helper.reconstruct_idexp_lm3d(sample['id'], sample['exp']).reshape([bs, x_mask.shape[1], -1])
                losses_out['mse_idexp_lm3d'] = self.lm468_mse_loss(gt_idexp_lm3d, pred_idexp_lm3d, x_mask)
                losses_out['l2_reg_id'] = self.l2_reg_loss(id_pred, x_mask)
                losses_out['l2_reg_exp'] = self.l2_reg_loss(exp_pred, x_mask)

                gt_lm2d = self.face3d_helper.reconstruct_lm2d(sample['id'], sample['exp'], sample['euler'], sample['trans']).reshape([bs, x_mask.shape[1], -1])
                pred_lm2d = self.face3d_helper.reconstruct_lm2d(id_pred, exp_pred, sample['euler'], sample['trans']).reshape([bs, x_mask.shape[1], -1])
                losses_out['mse_lm2d'] = self.lm468_mse_loss(gt_lm2d, pred_lm2d, x_mask)

            elif hparams["motion_type"] == 'exp':
                x_gt = sample['exp']
                sample['y'] = x_gt
                self.model(sample, model_out, train=True)
                x_pred = model_out['pred'].reshape([bs, -1, 64])
                x_mask = model_out['mask'].reshape([bs, -1])
                losses_out['kl'] = model_out['loss_kl']
                exp_pred = x_pred[:, :, :]
                losses_out['lap_exp'] = self.lap_loss(exp_pred, x_mask)
                if hparams.get("ref_id_mode",'first_frame') == 'first_frame':
                    id_pred = sample['id'][:,0:1, :].repeat([1,exp_pred.shape[1],1])
                elif hparams.get("ref_id_mode",'first_frame') == 'random_frame':
                    max_y = x_mask.sum(dim=1).min().item()
                    idx = random.randint(0, max_y-1)
                    id_pred = sample['id'][:,idx:idx+1, :].repeat([1,exp_pred.shape[1],1])
                gt_idexp_lm3d = self.face3d_helper.reconstruct_idexp_lm3d(sample['id'], sample['exp']).reshape([bs, x_mask.shape[1], -1])
                pred_idexp_lm3d = self.face3d_helper.reconstruct_idexp_lm3d(id_pred, exp_pred).reshape([bs, x_mask.shape[1], -1])
                losses_out['mse_exp'] = self.mse_loss(x_gt, x_pred, x_mask)
                losses_out['mse_idexp_lm3d'] = self.lm468_mse_loss(gt_idexp_lm3d, pred_idexp_lm3d, x_mask)
                losses_out['l2_reg_exp'] = self.l2_reg_loss(exp_pred, x_mask)
            
                gt_lm2d = self.face3d_helper.reconstruct_lm2d(sample['id'], sample['exp'], sample['euler'], sample['trans']).reshape([bs, x_mask.shape[1], -1])
                pred_lm2d = self.face3d_helper.reconstruct_lm2d(id_pred, exp_pred, sample['euler'], sample['trans']).reshape([bs, x_mask.shape[1], -1])
                # losses_out['mse_lm2d'] = self.lm468_mse_loss(gt_lm2d, pred_lm2d, x_mask)

            # calculating sync score
            mouth_lm3d = pred_idexp_lm3d.reshape([bs, x_pred.shape[1], 468*3]) # [b, t, 60]
            # mouth_lm3d = pred_idexp_lm3d.reshape([bs, x_pred.shape[1], 468, 3])[:,:, index_lm68_from_lm478,:][:,:,48:68].reshape([bs, x_pred.shape[1], 20*3]) # [b, t, 60]
            
            if hparams['audio_type'] == 'hubert':
                mel = sample['hubert'] # [b, 2*t, 1024]
            elif hparams['audio_type'] == 'mfcc':
                mel = sample['mfcc'] / 100 # [b, 2*t, 1024]
            elif hparams['audio_type'] == 'mel':
                mel = sample['mel'] # [b, 2*t, 1024]

            num_clips_for_syncnet = 8096
            len_mouth_slice = 5
            len_mel_slice = len_mouth_slice * 2
            num_iters = max(1, num_clips_for_syncnet // len(mouth_lm3d))
            mouth_clip_lst = []
            mel_clip_lst = []
            x_mask_clip_lst = []
            for i in range(num_iters):
                t_start = random.randint(0, x_pred.shape[1]-len_mouth_slice-1)
                mouth_clip = mouth_lm3d[:, t_start: t_start+len_mouth_slice]
                x_mask_clip = x_mask[:, t_start: t_start+len_mouth_slice]
                assert mouth_clip.shape[1] == len_mouth_slice
                mel_clip = mel[:, t_start*2 : t_start*2+len_mel_slice]
                mouth_clip_lst.append(mouth_clip)
                mel_clip_lst.append(mel_clip)
                x_mask_clip_lst.append(x_mask_clip)
            mouth_clips = torch.cat(mouth_clip_lst) # [B=8096, T=5, 60]
            mel_clips = torch.cat(mel_clip_lst) #  # [B=8096, T=10, 1024]
            x_mask_clips = torch.cat(x_mask_clip_lst) # [B=8096, T=5]
            x_mask_clips = (x_mask_clips.sum(dim=1) == x_mask_clips.shape[1]).float() # [B=8096,]
            audio_embedding, mouth_embedding = self.syncnet.forward(mel_clips, mouth_clips) # get normalized embedding, [B,]
            sync_loss, _ = self.syncnet.cal_sync_loss(audio_embedding, mouth_embedding, 1., reduction='none') # 
            losses_out['sync_lip_lm3d'] = (sync_loss * x_mask_clips).sum() / x_mask_clips.sum()
            return losses_out, model_out
    
    def kl_annealing(self, num_updates, max_lambda=0.4, t1=2000, t2=2000):
        """
        Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing
        https://aclanthology.org/N19-1021.pdf
        """
        T = t1 + t2
        num_updates = num_updates % T
        if num_updates < t1:
            return num_updates / t1 * max_lambda
        else:
            return max_lambda

    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'kl': self.kl_annealing(self.global_step, max_lambda=hparams['lambda_kl'], t1=hparams['lambda_kl_t1'], t2=hparams['lambda_kl_t2']),
            'mse_exp': hparams.get("lambda_mse_exp", 0.1),
            'mse_idexp_lm3d': hparams.get("lambda_mse_lm3d", 1.),
            'lap_id': hparams.get("lambda_lap_id", 1.),
            'lap_exp': hparams.get("lambda_lap_exp", 1.),
            'l2_reg_id': hparams.get("lambda_l2_reg_id", 0.),
            'l2_reg_exp': hparams.get("lambda_l2_reg_exp", 0.0),
            'sync_lip_lm3d': hparams.get("lambda_sync_lm3d", 0.2),
            'mse_lm2d': hparams.get("lambda_mse_lm2d", 0.)
        }
        
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])

        return total_loss, loss_output

    def validation_start(self):
        pass

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False, sync_batch_size=10000)
        outputs = tensors_to_scalars(outputs)
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)
        
    #####################
    # Testing
    #####################
    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        """
        :param sample:
        :param batch_idx:
        :return:
        """
        outputs = {}
        outputs['losses'], model_out = self.run_model(sample, infer=True)
        pred_exp = model_out['pred']
        self.save_result(pred_exp,  "pred_exp_val" , self.gen_dir)
        if hparams['save_gt']:
            base_fn = f"gt_exp_val"
            self.save_result(sample['exp'],  base_fn , self.gen_dir)
        return outputs

    def test_end(self, outputs):
        pass

    @staticmethod
    def save_result(exp_arr, base_fname, gen_dir):
        exp_arr = convert_to_np(exp_arr)
        np.save(f"{gen_dir}/{base_fname}.npy", exp_arr)
    
    def get_grad(self, opt_idx):
        grad_dict = {
            'grad/model': get_grad_norm(self.model),
        }
        return grad_dict
    

    def lm468_mse_loss(self, proj_lan, gt_lan, x_mask):
        b,t,c= proj_lan.shape
        # [B, T, 68*3]
        loss = ((proj_lan - gt_lan) ** 2) * x_mask[:,:, None]
        loss = loss.reshape([b,t,468,-1])

        unmatch_mask = [93, 127, 132, 234, 323, 356, 361, 454]
        upper_eye = [161,160,159,158,157] + [388,387,386,385,384]
        eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7] + [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249]
        inner_lip = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
        outer_lip = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]

        weights = torch.ones_like(loss)
        weights[:, :, eye] = 3
        weights[:, :, upper_eye] = 20
        weights[:, :, inner_lip] = 5
        weights[:, :, outer_lip] = 5
        weights[:, :, unmatch_mask] = 0

        loss = loss.reshape([b,t,c])
        weights = weights.reshape([b,t,c])
        return (loss * weights).sum() / (x_mask.sum()*c)
    
    def lm68_mse_loss(self, proj_lan, gt_lan, x_mask):
        b,t,c= proj_lan.shape
        # [B, T, 68*3]
        loss = ((proj_lan - gt_lan) ** 2) * x_mask[:,:, None]
        loss = loss.reshape([b,t,68,3])

        weights = torch.ones_like(loss)
        weights[:, :, 36:48, :] = 5 # eye 12 points
        weights[:, :, -8:, :] =  5 # inner lip 8 points
        weights[:, :, 28:31, :] =  5 # nose 3 points
        loss = loss.reshape([b,t,c])
        weights = weights.reshape([b,t,c])
        return (loss * weights).sum() / (x_mask.sum()*c)
    
    def l2_reg_loss(self, x_pred, x_mask):
        # mean absolute error, l1 loss
        error = (x_pred ** 2) * x_mask[:,:, None]
        num_frame = x_mask.sum()
        return error.sum() / (num_frame * self.in_out_dim)

    def lap_loss(self, in_tensor, x_mask):
        # [b, t, c]
        b,t,c = in_tensor.shape
        in_tensor = F.pad(in_tensor, pad=(0,0,1,1))
        in_tensor = rearrange(in_tensor, "b t c -> (b c) t").unsqueeze(1) # [B*c, 1, t]
        lap_kernel = torch.Tensor((-0.5, 1.0, -0.5)).reshape([1,1,3]).float().to(in_tensor.device) # [1, 1, kw]
        out_tensor = F.conv1d(in_tensor, lap_kernel) # [B*C, 1, T]
        out_tensor = out_tensor.squeeze(1)
        out_tensor = rearrange(out_tensor, "(b c) t -> b t c", b=b, t=t)
        loss_lap = (out_tensor**2) * x_mask.unsqueeze(-1)
        return loss_lap.sum() / (x_mask.sum()*c)

    def mse_loss(self, x_gt, x_pred, x_mask):
        # mean squared error, l2 loss
        error = (x_pred - x_gt) * x_mask[:,:, None]
        num_frame = x_mask.sum()
        return (error ** 2).sum() / (num_frame * self.in_out_dim)
    
    def mae_loss(self, x_gt, x_pred, x_mask):
        # mean absolute error, l1 loss
        error = (x_pred - x_gt) * x_mask[:,:, None]
        num_frame = x_mask.sum()
        return error.abs().sum() / (num_frame * self.in_out_dim)
    
    def vel_loss(self, x_pred, x_mask):
        # mean squared error, l2 loss
        error = (x_pred[:, 1:] - x_pred[:, :-1]) * x_mask[:,1:, None]
        num_frame = x_mask.sum()
        return (error).abs().sum() / (num_frame * self.in_out_dim)
    
    def continuity_loss(self, x_gt, x_pred, x_mask):
        # continuity loss, borrowed from <FACIAL: Synthesizing Dynamic Talking Face with Implicit Attribute Learning>
        diff_x_pred = x_pred[:,1:] - x_pred[:,:-1]
        diff_x_gt = x_gt[:,1:] - x_gt[:,:-1]
        error = (diff_x_pred[:,:,:] - diff_x_gt[:,:,:]) * x_mask[:,1:,None]
        init_error = x_pred[:,0,:] - x_gt[:,0,:]
        num_frame = x_mask.sum()
        return (error.pow(2).sum() + init_error.pow(2).sum()) / (num_frame * self.in_out_dim)