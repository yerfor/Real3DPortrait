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
from tqdm import tqdm
import csv
from utils.commons.hparams import hparams, set_hparams
from utils.commons.meters import Timer
from data_util.face3d_helper import Face3DHelper
from utils.audio import librosa_wav2mfcc
from utils.commons.dataset_utils import collate_xd


class SyncNet_Dataset(Dataset):
    def __init__(self, prefix='train', data_dir=None):
        self.hparams = hparams
        self.db_key = prefix
        self.ds_path = self.hparams['binary_data_dir'] if data_dir is None else data_dir
        self.ds = None
        self.sizes = None
        self.x_maxframes = 200 # 50 video frames
        self.face3d_helper = Face3DHelper('deep_3drecon/BFM')
        self.x_multiply = 8

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
        x_len = len(item['hubert'])
        y_len = x_len // 2 # video is 25fps
        item['id'] = item['id'][:y_len]
        item['exp'] = item['exp'][:y_len]
        
        # randomly select a fixed-length clip
        start_frames = random.randint(0,  max(0, x_len - self.x_maxframes))
        start_frames = start_frames // 2 * 2
        item['hubert'] = item['hubert'][start_frames: start_frames + self.x_maxframes]
        item['id'] = item['id'][start_frames//2: start_frames//2 + self.x_maxframes//2]
        item['exp'] = item['exp'][start_frames//2: start_frames//2 + self.x_maxframes//2]
        return item


    def get_dataloader(self, batch_size=1, num_workers=0):
        loader = DataLoader(self, pin_memory=True,collate_fn=self.collater, batch_size=batch_size, num_workers=num_workers)
        return loader


    def collater(self, samples):
        if len(samples) == 0:
            return None
        x_len = max(s['hubert'].size(0) for s in samples)
        y_len = x_len // 2
        batch = {
            'item_id': [s['item_id'] for s in samples],
        }
        batch['hubert'] = collate_xd([s["hubert"] for s in samples], max_len=x_len, pad_idx=0) # [b, t_max_y, 64]
        batch['x_mask'] = (batch['hubert'].abs().sum(dim=-1) > 0).float() # [b, t_max_x]

        batch['id'] = collate_xd([s["id"] for s in samples], max_len=y_len, pad_idx=0) # [b, t_max, 1]
        batch['exp'] = collate_xd([s["exp"] for s in samples], max_len=y_len, pad_idx=0) # [b, t_max, 1]
        batch['y_mask'] = (batch['id'].abs().sum(dim=-1) > 0).float() # [b, t_max_y]
        return batch


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"

    ds = SyncNet_Dataset("train", 'data/binary/th1kh')
    dl = ds.get_dataloader()
    for b in tqdm(dl):
        pass
    