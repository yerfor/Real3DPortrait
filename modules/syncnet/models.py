import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math


class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class LossScale(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(LossScale, self).__init__()
        
        self.wC = nn.Parameter(torch.tensor(init_w))
        self.bC = nn.Parameter(torch.tensor(init_b))

class CLIPLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, audio_features, motion_features, logit_scale, clip_mask=None):
        logits_per_audio = logit_scale * audio_features @ motion_features.T # [b,c]
        logits_per_motion = logit_scale * motion_features @ audio_features.T # [b,c]
        if clip_mask is not None:
            logits_per_audio += clip_mask        
            logits_per_motion += clip_mask
        labels = torch.arange(logits_per_motion.shape[0]).to(logits_per_motion.device)
        motion_loss = F.cross_entropy(logits_per_motion, labels)
        audio_loss = F.cross_entropy(logits_per_audio, labels)
        clip_loss = (motion_loss + audio_loss) / 2
        ret = {
            "audio_loss": audio_loss,
            "motion_loss": motion_loss,
            "clip_loss": clip_loss
        }
        return ret
        
    def compute_sync_conf(self, audio_features, motion_features, return_matrix=False):
        logits_per_audio = audio_features @ motion_features.T # [b,c]
        if return_matrix:
            return logits_per_audio
        return logits_per_audio[range(len(audio_features)), range(len(audio_features))]

class LandmarkHubertSyncNet(nn.Module):
    def __init__(self, lm_dim=60, audio_dim=1024, num_layers_per_block=3, base_hid_size=128, out_dim=512):
        super(LandmarkHubertSyncNet, self).__init__()

        self.clip_loss_fn = CLIPLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) * 0
        self.logit_scale_2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) * 0
        self.logit_scale_max = math.log(1. / 0.01)

        # hubert = torch.rand(B, 1024, t=10)
        hubert_layers = [
            Conv1d(audio_dim, base_hid_size, kernel_size=3, stride=1, padding=1)
        ]

        hubert_layers.append(
            Conv1d(base_hid_size, base_hid_size, kernel_size=3, stride=1, padding=1),
        )
        hubert_layers += [
            Conv1d(base_hid_size, base_hid_size, kernel_size=3, stride=1, padding=1, residual=True) for _ in range(num_layers_per_block-1)
        ]

        hubert_layers.append(
            Conv1d(base_hid_size, 2*base_hid_size, kernel_size=3, stride=2, padding=1),
        )
        hubert_layers += [
            Conv1d(2*base_hid_size, 2*base_hid_size, kernel_size=3, stride=1, padding=1, residual=True) for _ in range(num_layers_per_block-1)
        ]

        hubert_layers.append(
            Conv1d(2*base_hid_size, 4*base_hid_size, kernel_size=3, stride=2, padding=1),
        )
        hubert_layers += [
            Conv1d(4*base_hid_size, 4*base_hid_size, kernel_size=3, stride=1, padding=1, residual=True) for _ in range(num_layers_per_block-1)
        ]

        hubert_layers += [
            Conv1d(4*base_hid_size, 4*base_hid_size, kernel_size=3, stride=1, padding=1),
            Conv1d(4*base_hid_size, 4*base_hid_size, kernel_size=3, stride=1, padding=0),
            Conv1d(4*base_hid_size, 4*base_hid_size, kernel_size=1, stride=1, padding=0),
            Conv1d(4*base_hid_size, out_dim, kernel_size=1, stride=1, padding=0),
        ]
        self.hubert_encoder = nn.Sequential(*hubert_layers)

        # mouth = torch.rand(B, 20*3, t=5)
        mouth_layers = [
            Conv1d(lm_dim, 96, kernel_size=3, stride=1, padding=1)
        ]

        mouth_layers.append(
            Conv1d(96, base_hid_size, kernel_size=3, stride=1, padding=1),
        )
        mouth_layers += [
            Conv1d(base_hid_size, base_hid_size, kernel_size=3, stride=1, padding=1, residual=True) for _ in range(num_layers_per_block-1)
        ]

        mouth_layers.append(
            Conv1d(base_hid_size, 2*base_hid_size, kernel_size=3, stride=2, padding=1),
        )
        mouth_layers += [
            Conv1d(2*base_hid_size, 2*base_hid_size, kernel_size=3, stride=1, padding=1, residual=True) for _ in range(num_layers_per_block-1)
        ]

        mouth_layers.append(
            Conv1d(2*base_hid_size, 4*base_hid_size, kernel_size=3, stride=1, padding=1),
        )
        mouth_layers += [
            Conv1d(4*base_hid_size, 4*base_hid_size, kernel_size=3, stride=1, padding=1, residual=True) for _ in range(num_layers_per_block-1)
        ]

        mouth_layers += [
            Conv1d(4*base_hid_size, 4*base_hid_size, kernel_size=3, stride=1, padding=1),
            Conv1d(4*base_hid_size, 4*base_hid_size, kernel_size=3, stride=1, padding=0),
            Conv1d(4*base_hid_size, 4*base_hid_size, kernel_size=1, stride=1, padding=0),
            Conv1d(4*base_hid_size, out_dim, kernel_size=1, stride=1, padding=0),
        ]
        self.mouth_encoder = nn.Sequential(*mouth_layers)

        self.lm_dim = lm_dim
        self.audio_dim = audio_dim
        self.logloss = nn.BCELoss()

    def forward(self, hubert, mouth_lm): 
        # hubert := (B, T=10, C=1024)
        # mouth_lm3d := (B, T=5, C=60)
        hubert = hubert.transpose(1,2)
        mouth_lm = mouth_lm.transpose(1,2)
        mouth_embedding = self.mouth_encoder(mouth_lm)
        audio_embedding = self.hubert_encoder(hubert)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        mouth_embedding = mouth_embedding.view(mouth_embedding.size(0), -1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        mouth_embedding = F.normalize(mouth_embedding, p=2, dim=1)
        return audio_embedding, mouth_embedding

    def cal_sync_loss(self, audio_embedding, mouth_embedding, label, reduction='none'):
        if isinstance(label, torch.Tensor): # finegrained label
            gt_d = label.float().view(-1).to(audio_embedding.device)
        else: # int to represent global label, 1 denotes positive, and 0 denotes negative, used when calculate sync loss for other models
            gt_d = (torch.ones([audio_embedding.shape[0]]) * label).float().to(audio_embedding.device) # int
        d = F.cosine_similarity(audio_embedding, mouth_embedding) # [B]
        loss = F.binary_cross_entropy(d.reshape([audio_embedding.shape[0],]), gt_d, reduction=reduction)
        return loss, d

    def cal_clip_loss(self, audio_embedding, mouth_embedding, clip_mask=None):
        # logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
        logit_scale = 1
        clip_ret = self.clip_loss_fn(audio_embedding, mouth_embedding, logit_scale, clip_mask=clip_mask)
        loss = clip_ret['clip_loss']
        return loss

    def cal_clip_loss_local(self, audio_embedding, mouth_embedding, clip_mask=None):
        # logit_scale = torch.clamp(self.logit_scale_2, max=self.logit_scale_max).exp()
        logit_scale = 1
        clip_ret = self.clip_loss_fn(audio_embedding, mouth_embedding, logit_scale, clip_mask=clip_mask)
        loss = clip_ret['clip_loss']
        return loss

    def compute_sync_conf(self, audio_embedding, mouth_embedding, return_matrix=False):
        # logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
        logit_scale = 1
        clip_ret = self.clip_loss_fn.compute_sync_conf(audio_embedding, mouth_embedding, return_matrix)
        return clip_ret

if __name__ == '__main__':
    syncnet = LandmarkHubertSyncNet(lm_dim=204)
    hubert = torch.rand(2, 10, 1024)
    lm = torch.rand(2, 5, 204)
    mel_embedding, exp_embedding = syncnet(hubert, lm)
    label = torch.tensor([1., 0.])
    loss = syncnet.cal_sync_loss(mel_embedding, exp_embedding, label)
    print(" ")