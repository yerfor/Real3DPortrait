import torch
import random 

from utils.commons.base_task import BaseTask
from utils.commons.dataset_utils import data_loader
from utils.commons.hparams import hparams
from utils.commons.tensor_utils import tensors_to_scalars
from utils.nn.schedulers import CosineSchedule, NoneSchedule
from utils.nn.model_utils import print_arch, num_params
from utils.commons.ckpt_utils import load_ckpt

from modules.syncnet.models import LandmarkHubertSyncNet
from tasks.os_avatar.dataset_utils.syncnet_dataset import SyncNet_Dataset
from data_util.face3d_helper import Face3DHelper
    

class ScheduleForSyncNet(NoneSchedule):
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        self.lr = constant_lr

        lr = self.lr * hparams['lr_decay_rate'] ** (num_updates // hparams['lr_decay_interval'])
        # lr = max(lr, 5e-6)
        lr = max(lr, 5e-5)
        self.optimizer.param_groups[0]['lr'] = lr
        return self.lr


class SyncNetTask(BaseTask):
    def __init__(self, hparams_=None):
        global hparams
        if hparams_ is not None:
            hparams = hparams_
        self.hparams = hparams

        super().__init__()
        self.dataset_cls = SyncNet_Dataset

    def on_train_start(self):
        for n, m in self.model.named_children():
            num_params(m, model_name=n)

    def build_model(self):
        if self.hparams is not None:
            hparams = self.hparams

        # lm_dim = 468*3 # lip part in idexp_lm3d
        self.face3d_helper = Face3DHelper(use_gpu=False, keypoint_mode='lm68')
        if hparams.get('syncnet_keypoint_mode', 'lip') == 'lip':
            lm_dim = 20*3 # lip part in idexp_lm3d
        elif hparams['syncnet_keypoint_mode'] == 'lm68':
            lm_dim = 68*3 # lip part in idexp_lm3d
        elif hparams['syncnet_keypoint_mode'] == 'centered_lip':
            lm_dim = 20*3 # lip part in idexp_lm3d
        elif hparams['syncnet_keypoint_mode'] == 'centered_lip2d':
            lm_dim = 20*2 # lip part in idexp_lm3d
        elif hparams['syncnet_keypoint_mode'] == 'lm468':
            lm_dim = 468*3 # lip part in idexp_lm3d
            self.face3d_helper = Face3DHelper(use_gpu=False, keypoint_mode='mediapipe')

        if hparams['audio_type'] == 'hubert':
            audio_dim = 1024 # hubert
        elif hparams['audio_type'] == 'mfcc':
            audio_dim = 13 # hubert
        elif hparams['audio_type'] == 'mel':
            audio_dim = 80 # hubert
        self.model = LandmarkHubertSyncNet(lm_dim, audio_dim, num_layers_per_block=hparams['syncnet_num_layers_per_block'], base_hid_size=hparams['syncnet_base_hid_size'], out_dim=hparams['syncnet_out_hid_size'])
        print_arch(self.model)

        if hparams.get('init_from_ckpt', '') != '': 
            ckpt_dir = hparams.get('init_from_ckpt', '')
            load_ckpt(self.model, ckpt_dir, model_name='model', strict=False)

        return self.model

    def build_optimizer(self, model):
        if self.hparams is not None:
            hparams = self.hparams

        self.optimizer = optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']))
        return optimizer

    # def build_scheduler(self, optimizer):
    #     return CosineSchedule(optimizer, hparams['lr'], warmup_updates=0, total_updates=40_0000)
    def build_scheduler(self, optimizer):
        return ScheduleForSyncNet(optimizer, hparams['lr'])

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

    ##########################
    # training and validation
    ##########################
    def run_model(self, sample, infer=False, batch_size=1024):
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
        if self.hparams is not None:
            hparams = self.hparams

        if sample is None or len(sample) == 0:
            return None

        model_out = {}
        if 'idexp_lm3d' not in sample:
            with torch.no_grad():
                b,t,_ = sample['exp'].shape
                idexp_lm3d = self.face3d_helper.reconstruct_idexp_lm3d(sample['id'], sample['exp']).reshape([b,t,-1,3])
        else:
            b,t,*_ = sample['idexp_lm3d'].shape
            idexp_lm3d = sample['idexp_lm3d']

        if hparams.get('syncnet_keypoint_mode', 'lip') == 'lip':
            mouth_lm = idexp_lm3d[:,:, 48:68,:].reshape([b, t, 20*3]) # [b, t, 60]
        elif hparams.get('syncnet_keypoint_mode', 'lip') == 'centered_lip':
            mouth_lm = idexp_lm3d[:,:, 48:68, ].reshape([b, t, 20, 3]) # [b, t, 60]
            mean_mouth_lm = self.face3d_helper.key_mean_shape[48:68]
            mouth_lm = mouth_lm / 10 + mean_mouth_lm.reshape([1, 1, 20, 3]) - mean_mouth_lm.reshape([1, 1, 20, 3]).mean(dim=-2) # to center
            mouth_lm = mouth_lm.reshape([b, t, 20*3]) * 10
        elif hparams.get('syncnet_keypoint_mode', 'lip') == 'centered_lip2d':
            mouth_lm = idexp_lm3d[:,:, 48:68, ].reshape([b, t, 20, 3]) # [b, t, 60]
            mean_mouth_lm = self.face3d_helper.key_mean_shape[48:68]
            mouth_lm = mouth_lm / 10 + mean_mouth_lm.reshape([1, 1, 20, 3]) - mean_mouth_lm.reshape([1, 1, 20, 3]).mean(dim=-2) # to center
            mouth_lm = mouth_lm[..., :2]
            mouth_lm = mouth_lm.reshape([b, t, 20*2]) * 10
        elif hparams['syncnet_keypoint_mode'] == 'lm68':
            mouth_lm = idexp_lm3d.reshape([b, t, 68*3])
        elif hparams['syncnet_keypoint_mode'] == 'lm468':
            mouth_lm = idexp_lm3d.reshape([b, t, 468*3])

        if hparams['audio_type'] == 'hubert':
            mel = sample['hubert'] # [b, 2t, 1024]
        elif hparams['audio_type'] == 'mfcc':
            mel = sample['mfcc'] / 100 # [b, 2t, 1024]
        elif hparams['audio_type'] == 'mel':
            mel = sample['mfcc'] # [b, 2t, 1024]

        y_mask = sample['y_mask']
        y_len = y_mask.sum(dim=1).min().item() # [B, T]

        len_mouth_slice = 5 # 5 frames denotes 0.2s, which is a appropriate length for sync check
        len_mel_slice = len_mouth_slice * 2
        if infer:
            phase_ratio_dict = {
                'pos' : 1.0,
            }
        else:
            phase_ratio_dict = {
                'pos' : 0.4,
                'neg_same_people_small_offset_ratio' : 0.3,
                'neg_same_people_large_offset_ratio' : 0.2,
                'neg_diff_people_random_offset_ratio': 0.1
            }
        mouth_lst, mel_lst, label_lst = [], [], []
        for phase_key, phase_ratio in phase_ratio_dict.items():
            num_samples = int(batch_size * phase_ratio)
            if phase_key == 'pos':
                phase_mel_lst = []
                phase_mouth_lst = []
                num_iters = max(1, num_samples // len(mouth_lm))
                for i in range(num_iters):
                    t_start = random.randint(0, y_len-len_mouth_slice-1)
                    phase_mouth = mouth_lm[:, t_start: t_start+len_mouth_slice]
                    assert phase_mouth.shape[1] == len_mouth_slice
                    phase_mel = mel[:, t_start*2 : t_start*2+len_mel_slice]
                    phase_mouth_lst.append(phase_mouth)
                    phase_mel_lst.append(phase_mel)
                phase_mouth = torch.cat(phase_mouth_lst)
                phase_mel = torch.cat(phase_mel_lst)
                mouth_lst.append(phase_mouth)
                mel_lst.append(phase_mel)
                label_lst.append(torch.ones([len(phase_mel)])) # 1 denotes pos samples
            elif phase_key in ['neg_same_people_small_offset_ratio',  'neg_same_people_large_offset_ratio']:
                phase_mel_lst = []
                phase_mouth_lst = []
                num_iters = max(1, num_samples // len(mouth_lm))
                for i in range(num_iters):
                    if phase_key == 'neg_same_people_small_offset_ratio':
                        offset = random.choice([random.randint(-5,-2), random.randint(2,5)])
                    elif phase_key == 'neg_same_people_large_offset_ratio':
                        offset = random.choice([random.randint(-10,-5), random.randint(5,10)])
                    else: ValueError()

                    if offset < 0:
                        t_start = random.randint(-offset, y_len-len_mouth_slice-1)
                    else:
                        t_start = random.randint(0, y_len-len_mouth_slice-1-offset)
                    phase_mouth = mouth_lm[:, t_start: t_start+len_mouth_slice]
                    assert phase_mouth.shape[1] == len_mouth_slice
                    phase_mel = mel[:, (t_start+offset)*2:(t_start+offset)*2+len_mel_slice]
                    phase_mouth_lst.append(phase_mouth)
                    phase_mel_lst.append(phase_mel)
                phase_mouth = torch.cat(phase_mouth_lst)
                phase_mel = torch.cat(phase_mel_lst)
                mouth_lst.append(phase_mouth)
                mel_lst.append(phase_mel)
                label_lst.append(torch.zeros([len(phase_mel)])) # 0 denotes neg samples
            elif phase_key == 'neg_diff_people_random_offset_ratio':
                phase_mel_lst = []
                phase_mouth_lst = []
                num_iters = max(1, num_samples // len(mouth_lm))
                for i in range(num_iters):
                    offset = random.randint(-10, 10)
                    if offset < 0:
                        t_start = random.randint(-offset, y_len-len_mouth_slice-1)
                    else:
                        t_start = random.randint(0, y_len-len_mouth_slice-1-offset)
                    phase_mouth = mouth_lm[:, t_start: t_start+len_mouth_slice]
                    assert phase_mouth.shape[1] == len_mouth_slice
                    sample_idx = list(range(len(mouth_lm)))
                    random.shuffle(sample_idx)
                    phase_mel = mel[sample_idx, (t_start+offset)*2:(t_start+offset)*2+len_mel_slice]
                    phase_mouth_lst.append(phase_mouth)
                    phase_mel_lst.append(phase_mel)
                phase_mouth = torch.cat(phase_mouth_lst)
                phase_mel = torch.cat(phase_mel_lst)
                mouth_lst.append(phase_mouth)
                mel_lst.append(phase_mel)
                label_lst.append(torch.zeros([len(phase_mel)])) # 0 denotes neg samples

        mel_clips = torch.cat(mel_lst)
        mouth_clips = torch.cat(mouth_lst)
        labels = torch.cat(label_lst).float().to(mel_clips.device)

        audio_embedding, mouth_embedding = self.model(mel_clips, mouth_clips)
        sync_loss, cosine_sim = self.model.cal_sync_loss(audio_embedding, mouth_embedding, labels, reduction='mean')
        if not infer:
            losses_out = {}
            model_out = {}
            losses_out['sync_loss'] = sync_loss
            losses_out['batch_size'] = len(mel_clips)
            model_out['cosine_sim'] = cosine_sim
            return losses_out, model_out
        else:
            model_out['sync_loss'] = sync_loss
            model_out['batch_size'] = len(mel_clips)
            return model_out
            
    def _training_step(self, sample, batch_idx, optimizer_idx):
        ret = self.run_model(sample, infer=False, batch_size=hparams['syncnet_num_clip_pairs'])
        if ret is None:
            return None
        loss_output, model_out = ret
        loss_weights = {}
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        return total_loss, loss_output

    def validation_start(self):
        pass

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False, batch_size=8000)
        outputs = tensors_to_scalars(outputs)
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)
        
    #####################
    # Testing
    #####################
    def test_start(self):
        pass

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        """
        :param sample:
        :param batch_idx:
        :return:
        """
        pass

    def test_end(self, outputs):
        pass
