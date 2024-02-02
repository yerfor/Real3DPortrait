import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys

import glob
import cv2
import tqdm
import numpy as np
from data_gen.utils.mp_feature_extractors.face_landmarker import MediapipeLandmarker
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
import warnings
warnings.filterwarnings('ignore')

import random
random.seed(42)

import pickle
import json
import gzip
from typing import Any

def load_file(filename, is_gzip: bool = False, is_json: bool = False) -> Any:
    if is_json:
        if is_gzip:
            with gzip.open(filename, "r", encoding="utf-8") as f:
                loaded_object = json.load(f)
                return loaded_object
        else:
            with open(filename, "r", encoding="utf-8") as f:
                loaded_object = json.load(f)
                return loaded_object
    else:
        if is_gzip:
            with gzip.open(filename, "rb") as f:
                loaded_object = pickle.load(f)
                return loaded_object
        else:
            with open(filename, "rb") as f:
                loaded_object = pickle.load(f)
                return loaded_object
        
def save_file(filename, content, is_gzip: bool = False, is_json: bool = False) -> None:
    if is_json:
        if is_gzip:
            with gzip.open(filename, "w", encoding="utf-8") as f:
                json.dump(content, f)
        else:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(content, f)
    else:
        if is_gzip:
            with gzip.open(filename, "wb") as f:
                pickle.dump(content, f)
        else:
            with open(filename, "wb") as f:
                pickle.dump(content, f)

face_landmarker = None

def extract_lms_mediapipe_job(img):
    if img is None:
        return None
    global face_landmarker
    if face_landmarker is None:
        face_landmarker = MediapipeLandmarker()
    lm478 = face_landmarker.extract_lm478_from_img(img)
    return lm478
    
def extract_landmark_job(img_name):
    try:
        # if img_name == 'datasets/PanoHeadGen/raw/images/multi_view/chunk_0/seed0000002.png':
            # print(1)
            # input()
        out_name = img_name.replace("/images_512/", "/lms_2d/").replace(".png","_lms.npy")
        if os.path.exists(out_name):
            print("out exists, skip...")
            return
        try:
            os.makedirs(os.path.dirname(out_name), exist_ok=True)
        except:
            pass
        img = cv2.imread(img_name)[:,:,::-1]

        if img is not None:
            lm468 = extract_lms_mediapipe_job(img)
            if lm468 is not None:
                np.save(out_name, lm468)
        # print("Hahaha, solve one item!!!")
    except Exception as e:
        print(e)
        pass
        
def out_exist_job(img_name):
    out_name = img_name.replace("/images_512/", "/lms_2d/").replace(".png","_lms.npy") 
    if  os.path.exists(out_name):
        return None
    else:
        return img_name

# def get_todo_img_names(img_names):
#     todo_img_names = []
#     for i, res in multiprocess_run_tqdm(out_exist_job, img_names, num_workers=64):
#         if res is not None:
#             todo_img_names.append(res)
#     return todo_img_names


if __name__ == '__main__':
    import argparse, glob, tqdm, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default='/home/tiger/datasets/raw/FFHQ/images_512/')
    parser.add_argument("--ds_name", default='FFHQ')
    parser.add_argument("--num_workers", default=64, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--reset", action='store_true')
    parser.add_argument("--img_names_file", default="img_names.pkl", type=str)
    parser.add_argument("--load_img_names", action="store_true")

    args = parser.parse_args()
    print(f"args {args}")
    img_dir = args.img_dir
    img_names_file = os.path.join(img_dir, args.img_names_file)
    if args.load_img_names:
        img_names = load_file(img_names_file)
        print(f"load image names from {img_names_file}")
    else:
        if args.ds_name == 'FFHQ_MV':
            img_name_pattern1 = os.path.join(img_dir, "ref_imgs/*.png")
            img_names1 = glob.glob(img_name_pattern1)
            img_name_pattern2 = os.path.join(img_dir, "mv_imgs/*.png")
            img_names2 = glob.glob(img_name_pattern2)
            img_names = img_names1 + img_names2
            img_names = sorted(img_names)
        elif args.ds_name == 'FFHQ':
            img_name_pattern = os.path.join(img_dir, "*.png")
            img_names = glob.glob(img_name_pattern)
            img_names = sorted(img_names)
        elif args.ds_name == "PanoHeadGen":
            # img_name_patterns = ["ref/*/*.png", "multi_view/*/*.png", "reverse/*/*.png"]
            img_name_patterns = ["ref/*/*.png"]
            img_names = []
            for img_name_pattern in img_name_patterns:
                img_name_pattern_full = os.path.join(img_dir, img_name_pattern)
                img_names_part = glob.glob(img_name_pattern_full)
                img_names.extend(img_names_part)
            img_names = sorted(img_names)
        
    # save image names
    if not args.load_img_names:
        save_file(img_names_file, img_names)
        print(f"save image names in {img_names_file}")
        
    print(f"total images number: {len(img_names)}")
        
        
    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process -1
        num_samples_per_process = len(img_names) // total_process
        if process_id == total_process:
            img_names = img_names[process_id * num_samples_per_process : ]
        else:
            img_names = img_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
    
    # if not args.reset:
        # img_names = get_todo_img_names(img_names)
        

    print(f"todo_image {img_names[:10]}")
    print(f"processing images number in this process: {len(img_names)}")
    # print(f"todo images number: {len(img_names)}")
    # input()
    # exit()

    if args.num_workers == 1:
        index = 0
        for img_name in tqdm.tqdm(img_names, desc=f"Root process {args.process_id}: extracting MP-based landmark2d"):
            try:
                extract_landmark_job(img_name)
            except Exception as e:
                print(e)
                pass
            if index % max(1, int(len(img_names) * 0.003)) == 0:
                print(f"processed {index} / {len(img_names)}")
                sys.stdout.flush()
            index += 1
    else:
        for i, res in multiprocess_run_tqdm(
            extract_landmark_job, img_names, 
            num_workers=args.num_workers, 
            desc=f"Root {args.process_id}: extracing MP-based landmark2d"): 
            # if index % max(1, int(len(img_names) * 0.003)) == 0:
            print(f"processed {i+1} / {len(img_names)}")
            sys.stdout.flush()
        print(f"Root {args.process_id}: Finished extracting.")