import glob
import json
import os
import cv2
import pickle
import random
import re
import subprocess
from functools import partial

import librosa.core
import numpy as np
import torch
import torch.distributions
import torch.distributed as dist
import torch.optim
import torch.utils.data

from utils.commons.indexed_datasets import IndexedDataset
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import csv
from utils.commons.hparams import hparams, set_hparams
from utils.commons.meters import Timer
from data_util.face3d_helper import Face3DHelper
from utils.audio import librosa_wav2mfcc
from utils.commons.dataset_utils import collate_xd
from utils.commons.tensor_utils  import convert_to_tensor
from data_gen.utils.process_video.extract_segment_imgs import decode_segmap_mask_from_image
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
from utils.commons.image_utils import load_image_as_uint8_tensor
from modules.eg3ds.camera_utils.pose_sampler import UnifiedCameraPoseSampler


def sample_idx(img_dir, num_frames):
    cnt = 0
    while True:
        cnt += 1
        if cnt > 1000:
            print(f"recycle for more than 1000 times, check this {img_dir}")
        idx = random.randint(0, num_frames-1)
        ret1 = find_img_name(img_dir, idx)
        if ret1 == 'None':
            continue
        ret2 = find_img_name(img_dir.replace("/gt_imgs/","/head_imgs/"), idx)
        if ret2 == 'None':
            continue
        ret3 = find_img_name(img_dir.replace("/gt_imgs/","/inpaint_torso_imgs/"), idx)
        if ret3 == 'None':
            continue
        ret4 = find_img_name(img_dir.replace("/gt_imgs/","/com_imgs/"), idx)
        if ret4 == 'None':
            continue
        return idx
    

def find_img_name(img_dir, idx):
    gt_img_fname = os.path.join(img_dir, format(idx, "05d") + ".jpg")
    if not os.path.exists(gt_img_fname):
        gt_img_fname = os.path.join(img_dir, str(idx) + ".jpg")
    if not os.path.exists(gt_img_fname):
        gt_img_fname = os.path.join(img_dir, format(idx, "08d") + ".jpg")
    if not os.path.exists(gt_img_fname):
        gt_img_fname = os.path.join(img_dir, format(idx, "08d") + ".png")
    if not os.path.exists(gt_img_fname):
        gt_img_fname = os.path.join(img_dir, format(idx, "05d") + ".png")
    if not os.path.exists(gt_img_fname):
        gt_img_fname = os.path.join(img_dir, str(idx) + ".png")
    if os.path.exists(gt_img_fname):
        return gt_img_fname
    else:
        return 'None'
    
    
def get_win_from_arr(arr, index, win_size):
    left = index - win_size//2
    right = index + (win_size - win_size//2)
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > arr.shape[0]:
        pad_right = right - arr.shape[0]
        right = arr.shape[0]
    win = arr[left:right]
    if pad_left > 0:
        if isinstance(arr, np.ndarray):
            win = np.concatenate([np.zeros_like(win[:pad_left]), win], axis=0)
        else:
            win = torch.cat([torch.zeros_like(win[:pad_left]), win], dim=0)
    if pad_right > 0:
        if isinstance(arr, np.ndarray):
            win = np.concatenate([win, np.zeros_like(win[:pad_right])], axis=0) # [8, 16]
        else:
            win = torch.cat([win, torch.zeros_like(win[:pad_right])], dim=0) # [8, 16]
    return win


class Img2Plane_Dataset(Dataset):
    def __init__(self, prefix='train', data_dir=None):
        self.db_key = prefix
        self.ds = None
        self.sizes = None
        self.x_maxframes = 200 # 50 video frames
        self.face3d_helper = Face3DHelper('deep_3drecon/BFM')
        self.x_multiply = 8
        self.hparams = hparams
        self.pose_sampler = UnifiedCameraPoseSampler()
        self.ds_path = self.hparams['binary_data_dir'] if data_dir is None else data_dir

    def __len__(self):
        ds = self.ds = IndexedDataset(f'{self.ds_path}/{self.db_key}')
        return len(ds)

    def _get_item(self, index):
        """
        This func is necessary to open files in multi-threads!
        """
        if self.ds is None:
            self.ds = IndexedDataset(f'{self.ds_path}/{self.db_key}')
        return self.ds[index]
    
    def __getitem__(self, idx):
        raw_item = self._get_item(idx)
        if raw_item is None:
            print("loading from binary data failed!")
            return None
        item = {
            'idx': idx,
            'item_name': raw_item['img_dir'],
        }
        img_dir = raw_item['img_dir'].replace('/com_imgs/', '/gt_imgs/')
        num_frames = len(raw_item['exp'])

        hparams = self.hparams
        camera_ret = get_eg3d_convention_camera_pose_intrinsic({'euler':convert_to_tensor(raw_item['euler']).cpu(), 'trans':convert_to_tensor(raw_item['trans']).cpu()})
        c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
        raw_item['c2w'] = c2w
        raw_item['intrinsics'] = intrinsics


        max_pitch = 10 / 180 * 3.1415926 # range for mv pitch angle is smaller than that of ref
        min_pitch = -max_pitch
        pitch = random.random() * (max_pitch - min_pitch) + min_pitch
        max_yaw = 16 / 180 * 3.1415926
        min_yaw = - max_yaw
        yaw = random.random() * (max_yaw - min_yaw) + min_yaw
        distance = random.random() * (3.2-2.7) + 2.7 # [2.7, 4.0]
        ws_camera = self.pose_sampler.get_camera_pose(pitch, yaw, lookat_location=torch.tensor([0,0,0.2]), distance_to_orig=distance)[0]

        if hparams.get("random_sample_pose", False) is True and random.random() < 0.5 :
            max_pitch = 26 / 180 * 3.1415926 # range for mv pitch angle is smaller than that of ref
            min_pitch = -max_pitch
            pitch = random.random() * (max_pitch - min_pitch) + min_pitch
            max_yaw = 38 / 180 * 3.1415926
            min_yaw = - max_yaw
            yaw = random.random() * (max_yaw - min_yaw) + min_yaw
            distance = random.random() * (4.0-2.7) + 2.7 # [2.7, 4.0]
            real_camera = self.pose_sampler.get_camera_pose(pitch, yaw, lookat_location=torch.tensor([0,0,0.2]), distance_to_orig=distance)[0]
        else:
            real_idx = sample_idx(img_dir, num_frames)
            real_c2w = raw_item['c2w'][real_idx]
            real_intrinsics = raw_item['intrinsics'][real_idx]
            real_camera = np.concatenate([real_c2w.reshape([16,]) , real_intrinsics.reshape([9,])], axis=0)
            real_camera = convert_to_tensor(real_camera)

        if hparams.get("random_sample_pose", False) is True and random.random() < 0.5 :
            max_pitch = 26 / 180 * 3.1415926 # range for mv pitch angle is smaller than that of ref
            min_pitch = -max_pitch
            pitch = random.random() * (max_pitch - min_pitch) + min_pitch
            max_yaw = 38 / 180 * 3.1415926
            min_yaw = - max_yaw
            yaw = random.random() * (max_yaw - min_yaw) + min_yaw
            distance = random.random() * (4.0-2.7) + 2.7 # [2.7, 4.0]
            fake_camera = self.pose_sampler.get_camera_pose(pitch, yaw, lookat_location=torch.tensor([0,0,0.2]), distance_to_orig=distance)[0]
        else:
            fake_idx = sample_idx(img_dir, num_frames)
            fake_c2w = raw_item['c2w'][fake_idx]
            fake_intrinsics = raw_item['intrinsics'][fake_idx]
            fake_camera = np.concatenate([fake_c2w.reshape([16,]), fake_intrinsics.reshape([9,])], axis=0)
            fake_camera = convert_to_tensor(fake_camera)

        item.update({
            'ws_camera': ws_camera,
            'real_camera': real_camera,
            'fake_camera': fake_camera,
            # id,exp,euler,trans, used to generate the secc map
        })

        return item
    
    def get_dataloader(self, batch_size=1, num_workers=0):
        loader = DataLoader(self, pin_memory=True,collate_fn=self.collater, batch_size=batch_size, num_workers=num_workers)
        return loader

    def collater(self, samples):
        hparams = self.hparams
        if len(samples) == 0:
            return {}
        batch = {}

        batch['ffhq_ws_cameras'] = torch.stack([s['ws_camera'] for s in samples], dim=0) # [B, 204]
        batch['ffhq_ref_cameras'] = torch.stack([s['real_camera'] for s in samples], dim=0) # [B, 204]
        batch['ffhq_mv_cameras'] = torch.stack([s['fake_camera'] for s in samples], dim=0) # [B, 204]
        return batch



class Motion2Video_Dataset(Dataset):
    def __init__(self, prefix='train', data_dir=None):
        self.db_key = prefix
        self.ds = None
        self.sizes = None
        self.x_maxframes = 200 # 50 video frames
        self.face3d_helper = Face3DHelper('deep_3drecon/BFM')
        self.x_multiply = 8
        self.hparams = hparams
        self.ds_path = self.hparams['binary_data_dir'] if data_dir is None else data_dir

    def __len__(self):
        ds = self.ds = IndexedDataset(f'{self.ds_path}/{self.db_key}')
        return len(ds)

    def _get_item(self, index):
        """
        This func is necessary to open files in multi-threads!
        """
        if self.ds is None:
            self.ds = IndexedDataset(f'{self.ds_path}/{self.db_key}')
        return self.ds[index]
    
    def __getitem__(self, idx):
        raw_item = self._get_item(idx)
        if raw_item is None:
            print("loading from binary data failed!")
            return None
        item = {
            'idx': idx,
            'item_name': raw_item['img_dir'],
        }
            
        camera_ret = get_eg3d_convention_camera_pose_intrinsic({'euler':convert_to_tensor(raw_item['euler']).cpu(), 'trans':convert_to_tensor(raw_item['trans']).cpu()})
        c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
        raw_item['c2w'] = c2w
        raw_item['intrinsics'] = intrinsics

        img_dir = raw_item['img_dir'].replace('/com_imgs/', '/gt_imgs/')
        num_frames = len(raw_item['exp'])

        # src 
        real_idx = sample_idx(img_dir, num_frames)
        real_c2w = raw_item['c2w'][real_idx]
        
        real_intrinsics = raw_item['intrinsics'][real_idx]
        real_camera = np.concatenate([real_c2w.reshape([16,]) , real_intrinsics.reshape([9,])], axis=0)
        real_camera = convert_to_tensor(real_camera)
        item['real_camera'] = real_camera

        gt_img_fname = find_img_name(img_dir, real_idx)
        gt_img = load_image_as_uint8_tensor(gt_img_fname)[..., :3] # ignore alpha channel when png
        item['real_gt_img'] = gt_img.float() / 127.5 - 1
        # for key in ['head', 'torso', 'torso_with_bg', 'person']:
        for key in ['head', 'com', 'inpaint_torso']:
            key_img_dir = img_dir.replace("/gt_imgs/",f"/{key}_imgs/")
            key_img_fname = find_img_name(key_img_dir, real_idx)
            key_img = load_image_as_uint8_tensor(key_img_fname)[..., :3] # ignore alpha channel when png
            item[f'real_{key}_img'] = key_img.float() / 127.5 - 1
        bg_img_name = img_dir.replace("/gt_imgs/",f"/bg_img/") + '.jpg'
        bg_img = load_image_as_uint8_tensor(bg_img_name)[..., :3] # ignore alpha channel when png
        item[f'bg_img'] = bg_img.float() / 127.5 - 1

        seg_img_name = gt_img_fname.replace("/gt_imgs/",f"/segmaps/").replace(".jpg", ".png")
        seg_img = cv2.imread(seg_img_name)[:,:, ::-1]
        segmap = torch.from_numpy(decode_segmap_mask_from_image(seg_img)) # [6, H, W]
        item[f'real_segmap'] = segmap
        item[f'real_head_mask'] = segmap[[1,3,5]].sum(dim=0)
        item[f'real_torso_mask'] = segmap[[2,4]].sum(dim=0)
        item.update({
            # id,exp,euler,trans, used to generate the secc map
            'real_identity': convert_to_tensor(raw_item['id']).reshape([80,]),
            # 'real_identity': convert_to_tensor(raw_item['id'][real_idx]).reshape([80,]),
            'real_expression': convert_to_tensor(raw_item['exp'][real_idx]).reshape([64,]),
            'real_euler': convert_to_tensor(raw_item['euler'][real_idx]).reshape([3,]),
            'real_trans': convert_to_tensor(raw_item['trans'][real_idx]).reshape([3,]),
        })

        pertube_idx_candidates = [idx for idx in [real_idx-1,  real_idx+1] if (idx>=0 and idx <= num_frames-1 )] # previous frame
        # pertube_idx_candidates = [idx for idx in [real_idx-2,  real_idx-1,  real_idx+1,  real_idx+2] if (idx>=0 and idx <= num_frames-1 )] # previous frame
        pertube_idx = random.choice(pertube_idx_candidates)
        item[f'real_pertube_expression_1'] = convert_to_tensor(raw_item['exp'][pertube_idx]).reshape([64,])
        item[f'real_pertube_expression_2'] = item['real_expression'] * 2 - item[f'real_pertube_expression_1']

        # tgt
        fake_idx = sample_idx(img_dir, num_frames)
        min_offset = min(50, max((num_frames-1-fake_idx)//2, (fake_idx)//2))
        while abs(fake_idx - real_idx) < min_offset:
            fake_idx = sample_idx(img_dir, num_frames)
            min_offset = min(50, max((num_frames-1-fake_idx)//2, (fake_idx)//2))
        fake_c2w = raw_item['c2w'][fake_idx]

        fake_intrinsics = raw_item['intrinsics'][fake_idx]
        fake_camera = np.concatenate([fake_c2w.reshape([16,]) , fake_intrinsics.reshape([9,])], axis=0)
        fake_camera = convert_to_tensor(fake_camera)
        item['fake_camera'] = fake_camera
        
        gt_img_fname = find_img_name(img_dir, fake_idx)
        gt_img = load_image_as_uint8_tensor(gt_img_fname)[..., :3] # ignore alpha channel when png
        item['fake_gt_img'] = gt_img.float() / 127.5 - 1
        seg_img_name = gt_img_fname.replace("/gt_imgs/",f"/segmaps/").replace(".jpg", ".png")
        seg_img = cv2.imread(seg_img_name)[:,:, ::-1]
        segmap = torch.from_numpy(decode_segmap_mask_from_image(seg_img)) # [6, H, W]
        item[f'fake_segmap'] = segmap
        item[f'fake_head_mask'] = segmap[[1,3,5]].sum(dim=0)
        item[f'fake_torso_mask'] = segmap[[2,4]].sum(dim=0)
        # for key in ['head', 'torso', 'torso_with_bg', 'person']:
        for key in ['head', 'com', 'inpaint_torso']:
            key_img_dir = img_dir.replace("/gt_imgs/",f"/{key}_imgs/")
            key_img_fname = find_img_name(key_img_dir, fake_idx)
            key_img = load_image_as_uint8_tensor(key_img_fname)[..., :3] # ignore alpha channel when png
            item[f'fake_{key}_img'] = key_img.float() / 127.5 - 1

        item.update({
            # id,exp,euler,trans, used to generate the secc map
            f'fake_identity': convert_to_tensor(raw_item['id']).reshape([80,]),
            # f'fake_identity': convert_to_tensor(raw_item['id'][fake_idx]).reshape([80,]),
            f'fake_expression': convert_to_tensor(raw_item['exp'][fake_idx]).reshape([64,]),
            f'fake_euler': convert_to_tensor(raw_item['euler'][fake_idx]).reshape([3,]),
            f'fake_trans': convert_to_tensor(raw_item['trans'][fake_idx]).reshape([3,]),
        })

        # pertube_idx_candidates = [idx for idx in [fake_idx-2,  fake_idx-1,  fake_idx+1,  fake_idx+2] if (idx>=0 and idx <= num_frames-1 )] # previous frame
        pertube_idx_candidates = [idx for idx in [fake_idx-1,  fake_idx+1] if (idx>=0 and idx <= num_frames-1 )] # previous frame
        pertube_idx = random.choice(pertube_idx_candidates)
        item[f'fake_pertube_expression_1'] = convert_to_tensor(raw_item['exp'][pertube_idx]).reshape([64,])
        item[f'fake_pertube_expression_2'] = item['fake_expression'] * 2 - item[f'fake_pertube_expression_1']

        return item

    def get_dataloader(self, batch_size=1, num_workers=0):
        loader = DataLoader(self, pin_memory=True,collate_fn=self.collater, batch_size=batch_size, num_workers=num_workers)
        return loader

    def collater(self, samples):
        hparams = self.hparams
        if len(samples) == 0:
            return {}
        batch = {}

        batch['th1kh_item_names'] = [s['item_name'] for s in samples]
        batch['th1kh_ref_gt_imgs'] = torch.stack([s['real_gt_img'] for s in samples]).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]
        
        batch['th1kh_ref_head_masks'] = torch.stack([s['real_head_mask'] for s in samples]) # [B,6,H,W]
        batch['th1kh_ref_torso_masks'] = torch.stack([s['real_torso_mask'] for s in samples]) # [B,6,H,W]
        batch['th1kh_ref_segmaps'] = torch.stack([s['real_segmap'] for s in samples]) # [B,6,H,W]
        # for key in ['head', 'torso', 'torso_with_bg', 'person']:
        for key in ['head', 'com', 'inpaint_torso']:
            batch[f'th1kh_ref_{key}_imgs'] = torch.stack([s[f'real_{key}_img'] for s in samples]).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]
        batch[f'th1kh_bg_imgs'] = torch.stack([s[f'bg_img'] for s in samples]).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]
        
        batch['th1kh_ref_cameras'] = torch.stack([s['real_camera'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_ref_ids'] = torch.stack([s['real_identity'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_ref_exps'] = torch.stack([s['real_expression'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_ref_eulers'] = torch.stack([s['real_euler'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_ref_trans'] = torch.stack([s['real_trans'] for s in samples], dim=0) # [B, 204]

        batch['th1kh_mv_gt_imgs'] = torch.stack([s['fake_gt_img'] for s in samples]).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]
        # for key in ['head', 'torso', 'torso_with_bg', 'person']:
        for key in ['head', 'com', 'inpaint_torso']:
            batch[f'th1kh_mv_{key}_imgs'] = torch.stack([s[f'fake_{key}_img'] for s in samples]).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]

        batch['th1kh_mv_head_masks'] = torch.stack([s['fake_head_mask'] for s in samples]) # [B,6,H,W]
        batch['th1kh_mv_torso_masks'] = torch.stack([s['fake_torso_mask'] for s in samples]) # [B,6,H,W]
        batch['th1kh_mv_cameras'] = torch.stack([s['fake_camera'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_mv_ids'] = torch.stack([s['fake_identity'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_mv_exps'] = torch.stack([s['fake_expression'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_mv_eulers'] = torch.stack([s['fake_euler'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_mv_trans'] = torch.stack([s['fake_trans'] for s in samples], dim=0) # [B, 204]

        batch['th1kh_ref_pertube_exps_1'] = torch.stack([s['real_pertube_expression_1'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_ref_pertube_exps_2'] = torch.stack([s['real_pertube_expression_2'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_mv_pertube_exps_1'] = torch.stack([s['fake_pertube_expression_1'] for s in samples], dim=0) # [B, 204]
        batch['th1kh_mv_pertube_exps_2'] = torch.stack([s['fake_pertube_expression_2'] for s in samples], dim=0) # [B, 204]

        return batch

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"

    ds = Img2Plane_Dataset("train", 'data/binary/th1kh')
    # ds = Motion2Video_Dataset("train", 'data/binary/th1kh')
    dl = ds.get_dataloader()
    for b in tqdm(dl):
        pass
    