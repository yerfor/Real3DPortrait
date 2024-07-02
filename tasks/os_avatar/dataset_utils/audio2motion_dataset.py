import glob
import json
import os

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
import tqdm
import csv
from utils.commons.hparams import hparams, set_hparams
from utils.commons.meters import Timer
from data_util.face3d_helper import Face3DHelper
from utils.audio import librosa_wav2mfcc
from utils.commons.dataset_utils import collate_xd
from utils.commons.tensor_utils import convert_to_tensor

face3d_helper = None

def erosion_1d(arr):
    result = arr.copy()
    start_index = None
    continuous_length = 0

    for i, num in enumerate(arr):
        if num == 1:
            if continuous_length == 0:
                start_index = i
            continuous_length += 1
        else:
            if continuous_length > 0:
                # Replace middle 1s with 0s, keep first and last 1
                for j in range(start_index, start_index + continuous_length):
                    result[j] = 0
                result[start_index + continuous_length // 2] = 1
            continuous_length = 0
    if continuous_length > 0:
        # Replace middle 1s with 0s, keep first and last 1
        for j in range(start_index, start_index + continuous_length):
            result[j] = 0
        # result[start_index + continuous_length // 2] = 1
    return result

def get_mouth_amp(ldm):
    """
    ldm: [T, 68/468, 3]
    """
    is_mediapipe = ldm.shape[1] != 68 
    is_torch = isinstance(ldm, torch.Tensor)
    if not is_torch:
        ldm = torch.FloatTensor(ldm)
    if is_mediapipe:
        assert ldm.shape[1] in [468, 478]
        mouth_d = (ldm[:, 0] - ldm[:, 17]).abs().sum(-1)
    else:
        mouth_d = (ldm[:, 51] - ldm[:, 57]).abs().sum(-1)

    mouth_amp = torch.quantile(mouth_d, 0.9, dim=0)
    return mouth_amp

def get_eye_amp(ldm):
    """
    ldm: [T, 68/468, 3]
    """
    is_mediapipe = ldm.shape[1] != 68 
    is_torch = isinstance(ldm, torch.Tensor)
    if not is_torch:
        ldm = torch.FloatTensor(ldm)
    if is_mediapipe:
        assert ldm.shape[1] in [468, 478]
        eye_d = (ldm[:, 159] - ldm[:, 145]).abs().sum(-1) + (ldm[:, 386] - ldm[:, 374]).abs().sum(-1)
    else:
        eye_d = (ldm[:, 41] - ldm[:, 37]).abs().sum(-1) + (ldm[:, 40] - ldm[:, 38]).abs().sum(-1) + (ldm[:, 47] - ldm[:, 43]).abs().sum(-1) + (ldm[:, 46] - ldm[:, 44]).abs().sum(-1)

    eye_amp = torch.quantile(eye_d, 0.9, dim=0)
    return eye_amp

def get_blink(ldm):
    """
    ldm: [T, 68/468, 3]
    """
    is_mediapipe = ldm.shape[1] != 68 
    is_torch = isinstance(ldm, torch.Tensor)
    if not is_torch:
        ldm = torch.FloatTensor(ldm)
    if is_mediapipe:
        assert ldm.shape[1] in [468, 478]
        eye_d = (ldm[:, 159] - ldm[:, 145]).abs().sum(-1) + (ldm[:, 386] - ldm[:, 374]).abs().sum(-1)
    else:
        eye_d = (ldm[:, 41] - ldm[:, 37]).abs().sum(-1) + (ldm[:, 40] - ldm[:, 38]).abs().sum(-1) + (ldm[:, 47] - ldm[:, 43]).abs().sum(-1) + (ldm[:, 46] - ldm[:, 44]).abs().sum(-1)

    eye_d_qtl = torch.quantile(eye_d, 0.75, dim=0)
    blink = eye_d / eye_d_qtl
    blink = (blink < 0.85).long().numpy()
    blink = erosion_1d(blink)
    if is_torch:
        blink = torch.LongTensor(blink)
    return blink


class Audio2Motion_Dataset(Dataset):
    def __init__(self, prefix='train', data_dir=None):
        self.hparams = hparams
        self.db_key = prefix
        self.ds_path = self.hparams['binary_data_dir'] if data_dir is None else data_dir
        self.ds = None
        self.sizes = None
        self.x_maxframes = 200 # 50 video frames
        self.x_multiply = 8
        self.hparams = hparams

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
            'item_id': raw_item['img_dir'],
            'id': torch.from_numpy(raw_item['id']).float(), # [T_x, c=80]
            'exp': torch.from_numpy(raw_item['exp']).float(), # [T_x, c=80]
        }
        if item['id'].shape[0] == 1: # global_id
            item['id'] = item['id'].repeat([item['exp'].shape[0], 1])
        item['hubert'] = torch.from_numpy(raw_item['hubert']).float() # [T_x, 1024]
        item['f0'] = torch.from_numpy(raw_item['f0']).float() # [T_x,]

        global face3d_helper
        if face3d_helper is None:
            face3d_helper = Face3DHelper(use_gpu=False)
        cano_lm3d = face3d_helper.reconstruct_cano_lm3d(item['id'], item['exp'])
        item['blink_unit'] = get_blink(cano_lm3d)
        item['eye_amp'] = get_eye_amp(cano_lm3d)
        item['mouth_amp'] = get_mouth_amp(cano_lm3d)

        x_len = len(item['hubert'])
        x_len = x_len // self.x_multiply * self.x_multiply # make it divisible by our CNN
        y_len = x_len // 2 # video is 25fps
        item['hubert'] = item['hubert'][:x_len] # [T_x, c=80]
        item['f0'] = item['f0'][:x_len]
        
        item['id'] = item['id'][:y_len]
        item['exp'] = item['exp'][:y_len]
        item['euler'] = convert_to_tensor(raw_item['euler'][:y_len])
        item['trans'] = convert_to_tensor(raw_item['trans'][:y_len])
        item['blink_unit'] = item['blink_unit'][:y_len].reshape([-1,1])
        item['eye_amp'] = item['eye_amp'].reshape([1,])
        item['mouth_amp'] = item['mouth_amp'].reshape([1,])
        return item
    
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        sizes_fname = os.path.join(self.ds_path, f"sizes_{self.db_key}.npy")
        if os.path.exists(sizes_fname):
            sizes = np.load(sizes_fname, allow_pickle=True)
            self.sizes = sizes
        if self.sizes is None:
            self.sizes = []
            print("Counting the size of each item in dataset...")
            ds = IndexedDataset(f"{self.ds_path}/{self.db_key}")
            for i_sample in tqdm.trange(len(ds)):
                sample = ds[i_sample]
                if sample is None:
                    size = 0
                else:
                    x = sample['mel']
                    size = x.shape[-1] # time step in audio
                self.sizes.append(size)
            np.save(sizes_fname, self.sizes)
        indices = np.arange(len(self))
        indices = indices[np.argsort(np.array(self.sizes)[indices], kind='mergesort')]
        return indices

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1):
        """
        Yield mini-batches of indices bucketed by size. Batches may contain
        sequences of different lengths.

        Args:
            indices (List[int]): ordered list of dataset indices
            num_tokens_fn (callable): function that returns the number of tokens at
                a given index
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
        """
        def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            if len(batch) == 0:
                return 0
            if len(batch) == max_sentences:
                return 1
            if num_tokens > max_tokens:
                return 1
            return 0

        num_tokens_fn = lambda x: self.sizes[x]
        max_tokens = max_tokens if max_tokens is not None else 60000
        max_sentences = max_sentences if max_sentences is not None else 512
        bsz_mult = required_batch_size_multiple

        sample_len = 0
        sample_lens = []
        batch = []
        batches = []
        for i in range(len(indices)):
            idx = indices[i]
            num_tokens = num_tokens_fn(idx)
            sample_lens.append(num_tokens)
            sample_len = max(sample_len, num_tokens)

            assert sample_len <= max_tokens, (
                "sentence at index {} of size {} exceeds max_tokens "
                "limit of {}!".format(idx, sample_len, max_tokens)
            )
            num_tokens = (len(batch) + 1) * sample_len

            if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
                mod_len = max(
                    bsz_mult * (len(batch) // bsz_mult),
                    len(batch) % bsz_mult,
                )
                batches.append(batch[:mod_len])
                batch = batch[mod_len:]
                sample_lens = sample_lens[mod_len:]
                sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
            batch.append(idx)
        if len(batch) > 0:
            batches.append(batch)
        return batches
    

    def get_dataloader(self, batch_size=1, num_workers=0):
        batches_idx = self.batch_by_size(self.ordered_indices(), max_tokens=hparams['max_tokens_per_batch'], max_sentences=hparams['max_sentences_per_batch'])
        batches_idx = batches_idx * 50
        random.shuffle(batches_idx)
        loader = DataLoader(self, pin_memory=True,collate_fn=self.collater,  batch_size=batch_size, num_workers=num_workers)
        loader = DataLoader(self, pin_memory=True,collate_fn=self.collater,  batch_sampler=batches_idx, num_workers=num_workers)
        return loader

    def collater(self, samples):        
        hparams = self.hparams
        if len(samples) == 0:
            return {}

        batch = {}
        item_names = [s['item_id'] for s in samples]
        x_len = max(s['hubert'].size(0) for s in samples)
        assert x_len % self.x_multiply == 0
        y_len = x_len // 2

        batch['hubert'] = collate_xd([s["hubert"] for s in samples], max_len=x_len, pad_idx=0) # [b, t_max_y, 64]
        batch['x_mask'] = (batch['hubert'].abs().sum(dim=-1) > 0).float() # [b, t_max_x]
        batch['f0'] = collate_xd([s["f0"].reshape([-1,1]) for s in samples], max_len=x_len, pad_idx=0).squeeze(-1) # [b, t_max_y]

        batch.update({
            'item_id': item_names,
        })

        batch['id'] = collate_xd([s["id"] for s in samples], max_len=y_len, pad_idx=0) # [b, t_max, 1]
        batch['exp'] = collate_xd([s["exp"] for s in samples], max_len=y_len, pad_idx=0) # [b, t_max, 1]
        batch['euler'] = collate_xd([s["euler"] for s in samples], max_len=y_len, pad_idx=0) # [b, t_max, 1]
        batch['trans'] = collate_xd([s["trans"] for s in samples], max_len=y_len, pad_idx=0) # [b, t_max, 1]
        batch['blink_unit'] = collate_xd([s["blink_unit"] for s in samples], max_len=y_len, pad_idx=0) # [b, t_max, 1]
        batch['eye_amp'] = collate_xd([s["eye_amp"] for s in samples], max_len=1, pad_idx=0) # [b, t_max, 1]
        batch['mouth_amp'] = collate_xd([s["mouth_amp"] for s in samples], max_len=1, pad_idx=0) # [b, t_max, 1]
        batch['y_mask'] = (batch['id'].abs().sum(dim=-1) > 0).float() # [b, t_max_y]
        return batch


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    set_hparams('egs/os_avatar/audio2secc_vae.yaml')
    ds = Audio2Motion_Dataset("train", 'data/binary/th1kh')
    dl = ds.get_dataloader()
    for b in tqdm.tqdm(dl):
        pass
    