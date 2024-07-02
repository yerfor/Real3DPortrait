import os
import numpy as np
from scipy.misc import face
import torch
from tqdm import trange
import pickle
from copy import deepcopy

from data_util.face3d_helper import Face3DHelper
from utils.commons.indexed_datasets import IndexedDataset, IndexedDatasetBuilder


def load_video_npy(fn):
    assert fn.endswith("_coeff_fit_mp.npy")
    ret_dict = np.load(fn,allow_pickle=True).item()
    video_dict = {
        'euler': ret_dict['euler'], # [T, 3]
        'trans': ret_dict['trans'], # [T, 3]
        'id': ret_dict['id'], # [T, 80]
        'exp': ret_dict['exp'], # [T, 64]
    }
    return video_dict

def cal_lm3d_in_video_dict(video_dict, face3d_helper):
    identity = video_dict['id']
    exp = video_dict['exp']
    idexp_lm3d = face3d_helper.reconstruct_idexp_lm3d(identity, exp).cpu().numpy()
    video_dict['idexp_lm3d'] = idexp_lm3d


def load_audio_npy(fn):
    assert fn.endswith(".npy")
    ret_dict = np.load(fn,allow_pickle=True).item()
    audio_dict = {
        "mel": ret_dict['mel'], # [T, 80]
        "f0": ret_dict['f0'], # [T,1]
    }
    return audio_dict


if __name__ == '__main__':
    face3d_helper = Face3DHelper(use_gpu=False)
    
    import glob,tqdm
    prefixs = ['val', 'train']
    binarized_ds_path = "data/binary/th1kh"
    os.makedirs(binarized_ds_path, exist_ok=True)
    for prefix in prefixs:
        databuilder = IndexedDatasetBuilder(os.path.join(binarized_ds_path, prefix), gzip=False, default_idx_size=1024*1024*1024*2)
        raw_base_dir =  '/mnt/bn/ailabrenyi/entries/yezhenhui/datasets/raw/TH1KH_512/video'
        mp4_names = glob.glob(os.path.join(raw_base_dir, '*.mp4'))
        mp4_names = mp4_names[:1000]
        cnt = 0
        scnt = 0
        pbar = tqdm.tqdm(enumerate(mp4_names), total=len(mp4_names))
        for i, mp4_name in pbar:
            cnt += 1
            if prefix == 'train':
                if i % 100 == 0:
                    continue
            else:
                if i % 100 != 0:
                    continue
            hubert_npy_name = mp4_name.replace("/video/", "/hubert/").replace(".mp4", "_hubert.npy")
            audio_npy_name = mp4_name.replace("/video/", "/mel_f0/").replace(".mp4", "_mel_f0.npy")
            video_npy_name = mp4_name.replace("/video/", "/coeff_fit_mp/").replace(".mp4", "_coeff_fit_mp.npy")
            if not os.path.exists(audio_npy_name):
                print(f"Skip item for audio npy not found.")
                continue
            if not os.path.exists(video_npy_name):
                print(f"Skip item for video npy not found.")
                continue
            if (not os.path.exists(hubert_npy_name)):
                print(f"Skip item for hubert_npy not found.")
                continue
            audio_dict = load_audio_npy(audio_npy_name)
            hubert = np.load(hubert_npy_name)
            video_dict = load_video_npy(video_npy_name)
            com_img_dir = mp4_name.replace("/video/", "/com_imgs/").replace(".mp4", "")
            num_com_imgs = len(glob.glob(os.path.join(com_img_dir, '*')))
            num_frames = len(video_dict['exp'])
            if num_com_imgs != num_frames:
                print(f"Skip item for length mismatch.")
                continue
            mel = audio_dict['mel']
            if mel.shape[0] < 32: # the video is shorter than 0.6s
                print(f"Skip item for too short.")
                continue
            
            audio_dict.update(video_dict)
            audio_dict['item_id'] = os.path.basename(mp4_name)[:-4]
            audio_dict['hubert'] = hubert # [T_x, hid=1024]
            audio_dict['img_dir'] = com_img_dir


            databuilder.add_item(audio_dict)
            scnt += 1
            pbar.set_postfix({'success': scnt, 'success rate': scnt / cnt})
        databuilder.finalize()
        print(f"{prefix} set has {cnt} samples!")