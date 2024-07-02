#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.commons.hparams import hparams
import numpy as np
import math


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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float()
        res.append(correct_k)
    return res


class LossScale(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(LossScale, self).__init__()
        
        self.wC = nn.Parameter(torch.tensor(init_w))
        self.bC = nn.Parameter(torch.tensor(init_b))


class SyncNetModel(nn.Module):
    def __init__(self, auddim=1024, lipdim=20*3, nOut = 1024, stride=1):
        super(SyncNetModel, self).__init__()
        self.loss_scale = LossScale()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.clip_loss_fn = CLIPLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_max = math.log(1. / 0.01)

        self.netcnnaud = nn.Sequential(
            nn.Conv1d(auddim, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),

            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),

            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=(stride)),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),

            nn.Conv1d(512, 512, kernel_size=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, nOut, kernel_size=1),
        )

        self.netcnnlip = nn.Sequential(
            nn.Conv1d(lipdim, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),

            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 512, kernel_size=(3), padding=1, stride=(stride)),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),

            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, nOut, kernel_size=1),
        )

    def _forward_aud(self, x):
        # bct
        out = self.netcnnaud(x); # N x ch x 24 x M
        return out

    def _forward_vid(self, x):
        # bct
        out = self.netcnnlip(x); 
        return out
    
    def forward(self, hubert, mouth_lm): 
        # hubert := (B, T=100, C=1024)
        # mouth_lm3d := (B, T=50, C=60)
        # out: [B, T=50, C=1024]
        hubert = hubert.transpose(1,2)
        mouth_lm = mouth_lm.transpose(1,2)
        mouth_embedding = self._forward_vid(mouth_lm)
        audio_embedding = self._forward_aud(hubert)
        audio_embedding = audio_embedding.transpose(1,2)
        mouth_embedding = mouth_embedding.transpose(1,2)
        if hparams.get('normalize_embedding', False): # similar loss, no effects
            audio_embedding = F.normalize(audio_embedding, p=2, dim=-1)
            mouth_embedding = F.normalize(mouth_embedding, p=2, dim=-1)
        return audio_embedding.squeeze(1), mouth_embedding.squeeze(1)
    
    def _compute_sync_loss_batch(self, out_a, out_v, ymask=None):
        b, t, c = out_v.shape
        label = torch.arange(t).to(out_v.device)[None].repeat(b, 1)
        output = F.cosine_similarity(
            out_v[:, :, None], out_a[:, None, :], dim=-1) * self.loss_scale.wC + self.loss_scale.bC
        loss = self.criterion(output, label).mean()
        return loss
    
    def _compute_sync_loss(self, out_a, out_v, ymask=None):
         # b,t,c
        b, t, c = out_v.shape
        out_v = out_v.transpose(1,2)
        out_a = out_a.transpose(1,2)

        label = torch.arange(t).to(out_v.device)

        nloss = 0
        prec1 = 0
        if ymask is not None:
            total_num = ymask.sum()
        else:
            total_num = b*t
        for i in range(0, b):
            ft_v    = out_v[[i],:,:].transpose(2,0)
            ft_a    = out_a[[i],:,:].transpose(2,0)
            output  = F.cosine_similarity(ft_v, ft_a.transpose(0,2)) * self.loss_scale.wC + self.loss_scale.bC
            loss = self.criterion(output, label)
            if ymask is not None:
                loss = loss * ymask[i]
            nloss += loss.sum()
        nloss = nloss / total_num
        return nloss
    
    def compute_sync_loss(self,out_a, out_v, ymask=None, batch_mode=False):
        if batch_mode:
            return self._compute_sync_loss_batch(out_a, out_v)
        else:
            return self._compute_sync_loss(out_a, out_v)

    def compute_sync_score_for_infer(self, out_a, out_v, ymask=None):
         # b,t,c
        b, t, c = out_v.shape
        out_v = out_v.transpose(1,2)
        out_a = out_a.transpose(1,2)

        label = torch.arange(t).to(out_v.device)

        nloss = 0
        prec1 = 0
        if ymask is not None:
            total_num = ymask.sum()
        else:
            total_num = b*t
        for i in range(0, b):
            ft_v    = out_v[[i],:,:].transpose(2,0)
            ft_a    = out_a[[i],:,:].transpose(2,0)
            output  = F.cosine_similarity(ft_v, ft_a.transpose(0,2)) * self.loss_scale.wC + self.loss_scale.bC
            loss = self.criterion(output, label)
            if ymask is not None:
                loss = loss * ymask[i]
            nloss += loss.sum()
        nloss = nloss / total_num
        return nloss
        
    def cal_clip_loss(self, audio_embedding, mouth_embedding):
        logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
        clip_ret = self.clip_loss_fn(audio_embedding, mouth_embedding, logit_scale)
        loss = clip_ret['clip_loss']
        return loss

if __name__ == '__main__':
    syncnet = SyncNetModel()
    aud = torch.randn([2, 10, 1024])
    vid = torch.randn([2, 5, 60])
    aud_feat, vid_feat = syncnet.forward(aud, vid)

    print(aud_feat.shape)
    print(vid_feat.shape)