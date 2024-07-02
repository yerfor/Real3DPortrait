import json
import os
import cv2
import random
import copy
import random
import pickle
import tqdm
import numpy as np
import torch
import torch.distributions
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import traceback

from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
from utils.commons.image_utils import load_image_as_uint8_tensor
from tasks.eg3ds.dataset_utils.kv_dataset_base import BaseKVDataset, collate_xd, build_dataloader, KVSampler
from tasks.eg3ds.dataset_utils.video_reader import VideoReader
from modules.eg3ds.camera_utils.pose_sampler import UnifiedCameraPoseSampler
from tasks.eg3ds.dataset_utils.kv_eg3d_ffhq_dataset import flip_yaw_idexp_lm3d
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import decode_segmap_mask_from_image



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


class KV_FFHQ_Img2Plane_Dataset(BaseKVDataset):
    def __init__(self, prefix, shuffle=False, data_dir=None, max_len=None):
        self.hparams = hparams
        self.data_dir = self.hparams['binary_data_dir'] + '_kv' if data_dir is None else data_dir
        self.prefix = prefix
        self.num_parallel = 8
        self.num_fake_phase = 1
        ds_path = f'{self.data_dir}/data'
        super().__init__(ds_path, shuffle, num_parallel=self.num_parallel)
        if max_len is not None:
            self.key_and_sizes = self.key_and_sizes[:max_len]
        print(f"| {self.data_dir}.{prefix} dataset size:", len(self.key_and_sizes))

        self.pose_sampler = UnifiedCameraPoseSampler()
        self.segmenter = MediapipeSegmenter()

    def get_key_and_sizes(self):
        indexed_ds = self.get_reader()
        keys_path = f'{self.data_dir}/{self.prefix}_item_names.json'
        if self.prefix == 'test' or self.prefix == 'val':
            keys_path = f'{self.data_dir}/valid_item_names.json'
        if os.path.exists(keys_path):
            keys = json.load(open(keys_path))
        else:
            keys = indexed_ds.list_keys()
        keys = sorted(keys)
        return [(k, 1) for k in keys]
    
    def __getitem__(self, index):
        """
        index: list of int
        """
        hparams = self.hparams
        if self.indexed_ds is None:
            self.indexed_ds = self.get_reader(self.num_parallel)
        
        items = self.indexed_ds.read_many([str(self.key_and_sizes[idx][0]) for idx in index])
        fake_index = [random.randint(0, len(self)-1) for _ in range(self.num_fake_phase * len(items))]
        fake_items = self.indexed_ds.read_many([str(self.key_and_sizes[idx][0]) for idx in fake_index])

        samples = []
        for i_item, (item, i) in enumerate(zip(items, index)):
            item = pickle.loads(item)
            fake_item_lst = [pickle.loads(fake_it) for fake_it in fake_items[i_item*self.num_fake_phase: (i_item+1)*self.num_fake_phase]]
            sample = self.get_sample(i, item, fake_item_lst)
            if sample is not None:
                samples.append(sample)
        return samples
    
    def get_sample(self, id, raw_item, other_items):
        """
        item: raw_item obtained from indexed_ds
        """
        hparams = self.hparams
        item = {
            'item_id': id,
            'item_name': raw_item['item_name'],
        }

        img_name = raw_item['img_name']
        head_img_name = img_name.replace("/images_512/", "/head_imgs/").replace(".png", ".png")
        gt_img = load_image_as_uint8_tensor(img_name)[..., :3] # ignore alpha channel when png
        item[f'gt_img'] = gt_img.float() / 127.5 - 1
        head_img = load_image_as_uint8_tensor(head_img_name)[..., :3] # ignore alpha channel when png
        item[f'head_img'] = head_img.float() / 127.5 - 1

        real_c2w = raw_item['c2w']

        if hparams.get("random_sample_pose", False) is True and random.random() < 0.5 :
            max_pitch = 10 / 180 * 3.1415926 # range for mv pitch angle is smaller than that of ref
            min_pitch = -max_pitch
            pitch = random.random() * (max_pitch - min_pitch) + min_pitch
            max_yaw = 16 / 180 * 3.1415926
            min_yaw = - max_yaw
            yaw = random.random() * (max_yaw - min_yaw) + min_yaw
            distance = random.random() * (3.2-2.7) + 2.7 # [2.7, 4.0]
            ws_camera = self.pose_sampler.get_camera_pose(pitch, yaw, lookat_location=torch.tensor([0,0,0.2]), distance_to_orig=distance)[0]

            max_pitch = 26 / 180 * 3.1415926 # range for mv pitch angle is smaller than that of ref
            min_pitch = -max_pitch
            pitch = random.random() * (max_pitch - min_pitch) + min_pitch
            max_yaw = 38 / 180 * 3.1415926
            min_yaw = - max_yaw
            yaw = random.random() * (max_yaw - min_yaw) + min_yaw
            distance = random.random() * (4.0-2.7) + 2.7 # [2.7, 4.0]
            real_camera = self.pose_sampler.get_camera_pose(pitch, yaw, lookat_location=torch.tensor([0,0,0.2]), distance_to_orig=distance)[0]
        else:
            real_intrinsics = raw_item['intrinsics']
            real_camera = np.concatenate([real_c2w.reshape([16,]) , real_intrinsics.reshape([9,])], axis=0)
            real_camera = convert_to_tensor(real_camera)
            ws_camera = real_camera

        item.update({
            'ws_camera': ws_camera,
            'real_camera': real_camera,
            # id,exp,euler,trans, used to generate the secc map
            'real_identity': convert_to_tensor(raw_item['id']).reshape([80,]),
            'real_expression': convert_to_tensor(raw_item['exp']).reshape([64,]),
            'real_euler': convert_to_tensor(raw_item['euler']).reshape([3,]),
            'real_trans': convert_to_tensor(raw_item['trans']).reshape([3,]),
        })
        
        for i, other_item in enumerate(other_items):
            fake_idx = i + 1
            fake_c2w = other_item['c2w']

            if hparams.get("random_sample_pose", False) is True and random.random() < 0.5:
                max_pitch = 26 / 180 * 3.1415926 # range for mv pitch angle is smaller than that of ref
                min_pitch = -max_pitch
                pitch = random.random() * (max_pitch - min_pitch) + min_pitch
                max_yaw = 26 / 180 * 3.1415926
                min_yaw = - max_yaw
                yaw = random.random() * (max_yaw - min_yaw) + min_yaw
                distance = random.random() * (4.0-2.7) + 2.7 # [2.7, 4.0]
                fake_camera = self.pose_sampler.get_camera_pose(pitch, yaw, lookat_location=torch.tensor([0,0,0.2]), distance_to_orig=distance)[0]
            else:
                fake_intrinsics = other_item['intrinsics']
                fake_camera = np.concatenate([fake_c2w.reshape([16,]) , fake_intrinsics.reshape([9,])], axis=0)
                fake_camera = convert_to_tensor(fake_camera)
            item.update({
                f'fake_camera{fake_idx}': fake_camera,
                # id,exp,euler,trans, used to generate the secc map
                f'fake_identity{fake_idx}': convert_to_tensor(other_item['id']).reshape([80,]),
                f'fake_expression{fake_idx}': convert_to_tensor(other_item['exp']).reshape([64,]),
                f'fake_euler{fake_idx}': convert_to_tensor(other_item['euler']).reshape([3,]),
                f'fake_trans{fake_idx}': convert_to_tensor(other_item['trans']).reshape([3,]),
            })
        return item

    def collater(self, samples):
        hparams = self.hparams
        samples = samples[0]
        if len(samples) == 0:
            return {}
        batch = {}

        batch['ffhq_ws_cameras'] = torch.stack([s['ws_camera'] for s in samples], dim=0) # [B, 204]
        batch['ffhq_ref_cameras'] = torch.stack([s['real_camera'] for s in samples], dim=0) # [B, 204]
        batch['ffhq_mv_cameras'] = torch.stack([s['fake_camera1'] for s in samples], dim=0) # [B, 204]
        batch['ffhq_ref_ids'] = torch.stack([s['real_identity'] for s in samples], dim=0) # [B, 204]
        batch['ffhq_ref_exps'] = torch.stack([s['real_expression'] for s in samples], dim=0) # [B, 204]
        batch['ffhq_ref_eulers'] = torch.stack([s['real_euler'] for s in samples], dim=0) # [B, 204]
        batch['ffhq_ref_trans'] = torch.stack([s['real_trans'] for s in samples], dim=0) # [B, 204]
        return batch


class KV_TH1KH_Img2Plane_Dataset(BaseKVDataset):
    def __init__(self, prefix, shuffle=False, data_dir=None, max_len=None):
        self.hparams = hparams
        self.data_dir = self.hparams['binary_data_dir'] + '_kv' if data_dir is None else data_dir
        self.prefix = prefix
        self.num_parallel = 8
        ds_path = f'{self.data_dir}/data'
        super().__init__(ds_path, shuffle, num_parallel=self.num_parallel)
        if max_len is not None:
            self.key_and_sizes = self.key_and_sizes[:max_len]
        print(f"| {self.data_dir}.{prefix} dataset size:", len(self.key_and_sizes))

        with open(f'{self.data_dir}/spk_map.json', 'r') as f:
            spk_map = json.load(f)
            self.spk_name_to_spk_id = {spk_name : i for i, spk_name in enumerate(spk_map)}

    def get_key_and_sizes(self):
        indexed_ds = self.get_reader()
        keys_path = f'{self.data_dir}/{self.prefix}_item_names.json'
        if self.prefix == 'test' or self.prefix == 'val':
            keys_path = f'{self.data_dir}/valid_item_names.json'
        if os.path.exists(keys_path):
            keys = json.load(open(keys_path))
        else:
            keys = indexed_ds.list_keys()
        keys = sorted(keys)
        return [(k, 1) for k in keys]
    
    def __getitem__(self, index):
        """
        index: list of int
        """
        hparams = self.hparams
        if self.indexed_ds is None:
            self.indexed_ds = self.get_reader(self.num_parallel)
        
        items = self.indexed_ds.read_many([str(self.key_and_sizes[idx][0]) for idx in index])

        samples = []
        for i_item, (item, i) in enumerate(zip(items, index)):
            item = pickle.loads(item)
            try:
                sample = self.get_sample(i, item)
            except Exception as e:
                print(f"get_sample error: {e}")
                print(traceback.print_exc())
                sample = None
            if sample is not None:
                samples.append(sample)
        return samples
    
    def get_sample(self, id, raw_item):
        """
        item: raw_item obtained from indexed_ds
        """
        hparams = self.hparams
        item = {
            'item_id': id,
            'item_name': raw_item['item_name'],
            'spk_name': raw_item['spk_name'],
            'spk_id': self.spk_name_to_spk_id[raw_item['spk_name']],
        }

        img_dir = raw_item['img_dir'].replace('/com_imgs/', '/gt_imgs/')
        num_frames = raw_item['num_frames']

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

    def collater(self, samples):
        hparams = self.hparams
        samples = samples[0]
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


class KV_FFHQ_TH1KH_Img2Plane_Dataset(BaseKVDataset):
    def __init__(self, prefix, shuffle=False, data_dir=None, max_len=None):
        self.hparams = hparams
        self.ffhq_ds = KV_FFHQ_Img2Plane_Dataset(prefix, shuffle, data_dir='data/binary/FFHQ_kv')
        if hparams['ds_name'] == 'FFHQ_and_CelebV_HQ':
            self.th1kh_ds = KV_TH1KH_Img2Plane_Dataset(prefix, shuffle, data_dir='data/binary/CelebV-HQ_kv')
        elif hparams['ds_name'] == 'RAVDESS':
            self.th1kh_ds = KV_TH1KH_Img2Plane_Dataset(prefix, shuffle, data_dir=hparams["binary_data_dir"] + "_kv")
        else:
            self.th1kh_ds = KV_TH1KH_Img2Plane_Dataset(prefix, shuffle, data_dir='data/binary/TH1KH_512_kv')
        self.prefix = prefix
        self.len_ffhq = len(self.ffhq_ds)
        self.len_th1kh = len(self.th1kh_ds)
        self.sizes_ = [1 for _ in range(100000)]
        self.shuffle = shuffle

    def __getitem__(self, index):
        """
        index: list of int
        """
        hparams = self.hparams
        bs = len(index)
        ffhq_samples = self.ffhq_ds.__getitem__([random.randint(0, self.len_ffhq-1) for _ in range(bs)])
        th1kh_samples = self.th1kh_ds.__getitem__([random.randint(0, self.len_th1kh-1) for _ in range(bs)])
        samples = {}
        samples['ffhq_ws_camera'] = [s['ws_camera'] for s in ffhq_samples]
        samples['ffhq_ref_camera'] = [s['real_camera'] for s in ffhq_samples]
        samples['ffhq_mv_camera'] = [s['fake_camera1'] for s in ffhq_samples]
        samples['ffhq_ref_id'] = [s['real_identity'] for s in ffhq_samples]
        samples['ffhq_ref_exp'] = [s['real_expression'] for s in ffhq_samples]
        samples['ffhq_ref_euler'] = [s['real_euler'] for s in ffhq_samples]
        samples['ffhq_ref_trans'] = [s['real_trans'] for s in ffhq_samples]
        samples['ffhq_gt_img'] = [s['gt_img'] for s in ffhq_samples]
        samples['ffhq_head_img'] = [s['head_img'] for s in ffhq_samples]

        samples['th1kh_item_name'] = [s['item_name'] for s in th1kh_samples]

        samples['th1kh_ref_gt_img'] = [s['real_gt_img'] for s in th1kh_samples]
        samples['th1kh_ref_head_mask'] = [s['real_head_mask'] for s in th1kh_samples]
        samples['th1kh_ref_torso_mask'] = [s['real_torso_mask'] for s in th1kh_samples]
        samples['th1kh_ref_segmap'] = [s['real_segmap'] for s in th1kh_samples]
        # for key in ['head', 'torso', 'torso_with_bg', 'person']:
        for key in ['head', 'com', 'inpaint_torso']:
            samples[f'th1kh_ref_{key}_img'] = [s[f'real_{key}_img'] for s in th1kh_samples]
        samples[f'th1kh_bg_img'] = [s[f'bg_img'] for s in th1kh_samples]

        samples['th1kh_ref_camera'] = [s['real_camera'] for s in th1kh_samples]
        samples['th1kh_ref_id'] = [s['real_identity'] for s in th1kh_samples]
        samples['th1kh_ref_exp'] = [s['real_expression'] for s in th1kh_samples]
        samples['th1kh_ref_euler'] = [s['real_euler'] for s in th1kh_samples]
        samples['th1kh_ref_trans'] = [s['real_trans'] for s in th1kh_samples]

        samples['th1kh_mv_gt_img'] = [s['fake_gt_img'] for s in th1kh_samples]
        samples['th1kh_mv_head_mask'] = [s['fake_head_mask'] for s in th1kh_samples]
        samples['th1kh_mv_torso_mask'] = [s['fake_torso_mask'] for s in th1kh_samples]
        # for key in ['head', 'torso', 'torso_with_bg', 'person']:
        for key in ['head', 'com', 'inpaint_torso']:
            samples[f'th1kh_mv_{key}_img'] = [s[f'fake_{key}_img'] for s in th1kh_samples]

        samples['th1kh_mv_camera'] = [s['fake_camera'] for s in th1kh_samples]
        samples['th1kh_mv_id'] = [s['fake_identity'] for s in th1kh_samples]
        samples['th1kh_mv_exp'] = [s['fake_expression'] for s in th1kh_samples]
        samples['th1kh_mv_euler'] = [s['fake_euler'] for s in th1kh_samples]
        samples['th1kh_mv_trans'] = [s['fake_trans'] for s in th1kh_samples]

        samples['th1kh_ref_pertube_exp_1'] = [s['real_pertube_expression_1'] for s in th1kh_samples]
        samples['th1kh_ref_pertube_exp_2'] = [s['real_pertube_expression_2'] for s in th1kh_samples]
        samples['th1kh_mv_pertube_exp_1'] = [s['fake_pertube_expression_1'] for s in th1kh_samples]
        samples['th1kh_mv_pertube_exp_2'] = [s['fake_pertube_expression_2'] for s in th1kh_samples]

        return samples

    def collater(self, samples):
        hparams = self.hparams
        samples = samples[0]
        if len(samples) == 0:
            return {}
        batch = {}

        batch['ffhq_ws_cameras'] = torch.stack(samples['ffhq_ws_camera'], dim=0) # [B, 204]
        batch['ffhq_ref_cameras'] = torch.stack(samples['ffhq_ref_camera'], dim=0) # [B, 204]
        batch['ffhq_mv_cameras'] = torch.stack(samples['ffhq_mv_camera'], dim=0) # [B, 204]
        batch['ffhq_ref_ids'] = torch.stack(samples['ffhq_ref_id'], dim=0) # [B, 204]
        batch['ffhq_ref_exps'] = torch.stack(samples['ffhq_ref_exp'], dim=0) # [B, 204]
        batch['ffhq_ref_eulers'] = torch.stack(samples['ffhq_ref_euler'], dim=0) # [B, 204]
        batch['ffhq_ref_trans'] = torch.stack(samples['ffhq_ref_trans'], dim=0) # [B, 204]
        batch['ffhq_gt_imgs'] = torch.stack(samples['ffhq_gt_img']).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]
        batch['ffhq_head_imgs'] = torch.stack(samples['ffhq_head_img']).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]

        batch['th1kh_item_names'] = samples['th1kh_item_name']
        batch['th1kh_ref_gt_imgs'] = torch.stack(samples['th1kh_ref_gt_img']).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]
        batch['th1kh_ref_segmaps'] = torch.stack(samples['th1kh_ref_segmap']) # [B,6,H,W]
        batch['th1kh_ref_head_masks'] = torch.stack(samples['th1kh_ref_head_mask']) # [B,6,H,W]
        # for key in ['head', 'torso', 'torso_with_bg', 'person']:
        for key in ['head', 'com', 'inpaint_torso']:
            batch[f'th1kh_ref_{key}_imgs'] = torch.stack(samples[f'th1kh_ref_{key}_img']).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]
        batch[f'th1kh_bg_imgs'] = torch.stack(samples[f'th1kh_bg_img']).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]
        
        batch['th1kh_ref_cameras'] = torch.stack(samples['th1kh_ref_camera'], dim=0) # [B, 204]
        batch['th1kh_ref_ids'] = torch.stack(samples['th1kh_ref_id'], dim=0) # [B, 204]
        batch['th1kh_ref_exps'] = torch.stack(samples['th1kh_ref_exp'], dim=0) # [B, 204]
        batch['th1kh_ref_eulers'] = torch.stack(samples['th1kh_ref_euler'], dim=0) # [B, 204]
        batch['th1kh_ref_trans'] = torch.stack(samples['th1kh_ref_trans'], dim=0) # [B, 204]

        batch['th1kh_mv_gt_imgs'] = torch.stack(samples['th1kh_mv_gt_img']).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]
        batch['th1kh_mv_head_masks'] = torch.stack(samples['th1kh_mv_head_mask']) # [B,6,H,W]
        # for key in ['head', 'torso', 'torso_with_bg', 'person']:
        for key in ['head', 'com', 'inpaint_torso']:
            batch[f'th1kh_mv_{key}_imgs'] = torch.stack(samples[f'th1kh_mv_{key}_img']).permute(0,3,1,2) # [B, H, W, 3]==>[B,3,H,W]

        batch['th1kh_mv_cameras'] = torch.stack(samples['th1kh_mv_camera'], dim=0) # [B, 204]
        batch['th1kh_mv_ids'] = torch.stack(samples['th1kh_mv_id'], dim=0) # [B, 204]
        batch['th1kh_mv_exps'] = torch.stack(samples['th1kh_mv_exp'], dim=0) # [B, 204]
        batch['th1kh_mv_eulers'] = torch.stack(samples['th1kh_mv_euler'], dim=0) # [B, 204]
        batch['th1kh_mv_trans'] = torch.stack(samples['th1kh_mv_trans'], dim=0) # [B, 204]

        batch['th1kh_ref_pertube_exps_1'] = torch.stack(samples['th1kh_ref_pertube_exp_1'], dim=0) # [B, 204]
        batch['th1kh_ref_pertube_exps_2'] = torch.stack(samples['th1kh_ref_pertube_exp_2'], dim=0) # [B, 204]
        batch['th1kh_mv_pertube_exps_1'] = torch.stack(samples['th1kh_mv_pertube_exp_1'], dim=0) # [B, 204]
        batch['th1kh_mv_pertube_exps_2'] = torch.stack(samples['th1kh_mv_pertube_exp_2'], dim=0) # [B, 204]

        return batch

    def get_dataloader(self):
        hparams = self.hparams

        if self.prefix == 'train':
            use_ddp = False #  len(os.environ['CUDA_VISIBLE_DEVICES']) > 1
            dl = build_dataloader(self, shuffle=True, use_ddp=use_ddp, 
                                  max_sentences=hparams['batch_size'], 
                                  max_tokens=None, endless=True, 
                                  is_batch_by_size=False, chunked_read=True, 
                                  num_workers=hparams['num_workers'], training=True,
                                  prefetch_factor=4)
        else:
            dl = build_dataloader(self, shuffle=False, use_ddp=False, 
                                  max_sentences=hparams['batch_size'], 
                                  max_tokens=None, endless=False, 
                                  is_batch_by_size=False, chunked_read=True, 
                                  num_workers=hparams['num_workers'], training=False)
        return dl


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"

    set_hparams('egs/datasets/os_avatar/secc_img2plane.yaml')
    hparams['num_workers'] = 8
    ds = KV_FFHQ_TH1KH_Img2Plane_Dataset('train')
    dl = ds.get_dataloader()
    iterator = iter(dl)
    batch = iterator.__next__()
    false_cnt = 0
    min_trans, max_trans = 100, -100
    min_trans2, max_trans2 = 100, -100
    for j, i in tqdm.tqdm(enumerate(dl)):
        cam = i['ffhq_ref_cameras']
        cam2world = cam[:, :16].reshape([-1, 4,4])
        trans = cam2world[:, :3,3].reshape([-1, 3])
        trans = trans[:, -1]
        min_trans = min(trans.min().item(), min_trans)
        max_trans = max(trans.max().item(), max_trans)
        if j % 100 == 0:
            print(j, ":", min_trans, ",",max_trans)

        cam = i['th1kh_ref_cameras']
        cam2world = cam[:, :16].reshape([-1, 4,4])
        trans = cam2world[:, :3,3].reshape([-1, 3])
        trans = trans[:, -1]
        min_trans = min(trans.min().item(), min_trans2)
        max_trans = max(trans.max().item(), max_trans2)
        if j % 100 == 0:
            print(j, ":", min_trans, ",",max_trans)