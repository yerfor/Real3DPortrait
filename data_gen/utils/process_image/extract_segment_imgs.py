import os
os.environ["OMP_NUM_THREADS"] = "1"

import glob
import cv2
import tqdm
import numpy as np
import PIL
from utils.commons.tensor_utils import convert_to_np
import torch
import mediapipe as mp
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background, save_rgb_image_to_path
seg_model = MediapipeSegmenter()


def extract_segment_job(img_name):
    try:
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        segmap = seg_model._cal_seg_map(img)
        bg_img = extract_background([img], [segmap])
        out_img_name = img_name.replace("/images_512/",f"/bg_img/").replace(".mp4", ".jpg")
        save_rgb_image_to_path(bg_img, out_img_name)

        com_img = img.copy()
        bg_part = segmap[0].astype(bool)[..., None].repeat(3,axis=-1)
        com_img[bg_part] = bg_img[bg_part]
        out_img_name = img_name.replace("/images_512/",f"/com_imgs/")
        save_rgb_image_to_path(com_img, out_img_name)

        for mode in ['head', 'torso', 'person', 'torso_with_bg', 'bg']:
            out_img, _ = seg_model._seg_out_img_with_segmap(img, segmap, mode=mode)
            out_img_name = img_name.replace("/images_512/",f"/{mode}_imgs/")
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            try: os.makedirs(os.path.dirname(out_img_name), exist_ok=True)
            except: pass
            cv2.imwrite(out_img_name, out_img)

        inpaint_torso_img, inpaint_torso_with_bg_img, _, _ = inpaint_torso_job(img, segmap)
        out_img_name = img_name.replace("/images_512/",f"/inpaint_torso_imgs/")
        save_rgb_image_to_path(inpaint_torso_img, out_img_name)
        inpaint_torso_with_bg_img[bg_part] = bg_img[bg_part]
        out_img_name = img_name.replace("/images_512/",f"/inpaint_torso_with_com_bg_imgs/")
        save_rgb_image_to_path(inpaint_torso_with_bg_img, out_img_name)
        return 0
    except Exception as e:
        print(e)
        return 1

def out_exist_job(img_name):
    out_name1 = img_name.replace("/images_512/", "/head_imgs/")
    out_name2 = img_name.replace("/images_512/", "/com_imgs/")
    out_name3 = img_name.replace("/images_512/", "/inpaint_torso_with_com_bg_imgs/")
    
    if  os.path.exists(out_name1) and os.path.exists(out_name2) and os.path.exists(out_name3):
        return None
    else:
        return img_name

def get_todo_img_names(img_names):
    todo_img_names = []
    for i, res in multiprocess_run_tqdm(out_exist_job, img_names, num_workers=64):
        if res is not None:
            todo_img_names.append(res)
    return todo_img_names


if __name__ == '__main__':
    import argparse, glob, tqdm, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default='./images_512')
    # parser.add_argument("--img_dir", default='/home/tiger/datasets/raw/FFHQ/images_512')
    parser.add_argument("--ds_name", default='FFHQ')
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--reset", action='store_true')

    args = parser.parse_args()
    img_dir = args.img_dir
    if args.ds_name == 'FFHQ_MV':
        img_name_pattern1 = os.path.join(img_dir, "ref_imgs/*.png")
        img_names1 = glob.glob(img_name_pattern1)
        img_name_pattern2 = os.path.join(img_dir, "mv_imgs/*.png")
        img_names2 = glob.glob(img_name_pattern2)
        img_names = img_names1 + img_names2
    elif args.ds_name == 'FFHQ':
        img_name_pattern = os.path.join(img_dir, "*.png")
        img_names = glob.glob(img_name_pattern)
    
    img_names = sorted(img_names)
    random.seed(args.seed)
    random.shuffle(img_names)

    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process -1
        num_samples_per_process = len(img_names) // total_process
        if process_id == total_process:
            img_names = img_names[process_id * num_samples_per_process : ]
        else:
            img_names = img_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
    
    if not args.reset:
        img_names = get_todo_img_names(img_names)
    print(f"todo images number: {len(img_names)}")

    for vid_name in multiprocess_run_tqdm(extract_segment_job ,img_names, desc=f"Root process {args.process_id}: extracting segment images", num_workers=args.num_workers):
        pass