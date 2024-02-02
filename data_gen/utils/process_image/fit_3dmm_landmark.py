from numpy.core.numeric import require
from numpy.lib.function_base import quantile
import torch
import torch.nn.functional as F
import copy
import numpy as np

import os
import sys
import cv2
import argparse
import tqdm
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from data_gen.utils.mp_feature_extractors.face_landmarker import MediapipeLandmarker

from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
import pickle

face_model = ParametricFaceModel(bfm_folder='deep_3drecon/BFM', 
            camera_distance=10, focal=1015, keypoint_mode='mediapipe')
face_model.to("cuda")
     

index_lm68_from_lm468 = [127,234,93,132,58,136,150,176,152,400,379,365,288,361,323,454,356,70,63,105,66,107,336,296,334,293,300,168,197,5,4,75,97,2,326,305,
                         33,160,158,133,153,144,362,385,387,263,373,380,61,40,37,0,267,270,291,321,314,17,84,91,78,81,13,311,308,402,14,178]

dir_path = os.path.dirname(os.path.realpath(__file__))

LAMBDA_REG_ID = 0.3
LAMBDA_REG_EXP = 0.05

def save_file(name, content):
    with open(name, "wb") as f:
        pickle.dump(content, f) 
        
def load_file(name):
    with open(name, "rb") as f:
        content = pickle.load(f)
    return content

def cal_lan_loss_mp(proj_lan, gt_lan):
    # [B, 68, 2]
    loss = (proj_lan - gt_lan).pow(2)
    # loss = (proj_lan - gt_lan).abs()
    unmatch_mask = [ 93, 127, 132, 234, 323, 356, 361, 454]
    eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7] + [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249]
    inner_lip = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
    outer_lip = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
    weights = torch.ones_like(loss)
    weights[:, eye] = 5
    weights[:, inner_lip] = 2
    weights[:, outer_lip] = 2
    weights[:, unmatch_mask] = 0
    loss = loss * weights
    return torch.mean(loss)
 
def cal_lan_loss(proj_lan, gt_lan):
    # [B, 68, 2]
    loss = (proj_lan - gt_lan)** 2
    # use the ldm weights from deep3drecon, see deep_3drecon/deep_3drecon_models/losses.py
    weights = torch.zeros_like(loss)
    weights = torch.ones_like(loss)
    weights[:, 36:48, :] = 3 # eye 12 points
    weights[:, -8:, :] =  3 # inner lip 8 points
    weights[:, 28:31, :] =  3 # nose 3 points
    loss = loss * weights
    return torch.mean(loss)

def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True

def read_video_to_frames(img_name):
    frames = []
    cap = cv2.VideoCapture(img_name)
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    return np.stack(frames)
    
@torch.enable_grad()
def fit_3dmm_for_a_image(img_name, debug=False, keypoint_mode='mediapipe', device="cuda:0", save=True):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[0], img.shape[0]
    assert img_h == img_w
    num_frames = 1

    lm_name = img_name.replace("/images_512/", "/lms_2d/").replace(".png", "_lms.npy")
    if lm_name.endswith('_lms.npy') and os.path.exists(lm_name):
        lms = np.load(lm_name)
    else:
        # print("lms_2d file not found, try to extract it from image...")
        try:
            landmarker = MediapipeLandmarker()
            lms = landmarker.extract_lm478_from_img_name(img_name)
            # lms = landmarker.extract_lm478_from_img(img)
        except Exception as e:
            print(e)
            return
        if lms is None:
            print("get None lms_2d, please check whether each frame has one head, exiting...")
            return
    lms = lms[:468].reshape([468,2])
    lms = torch.FloatTensor(lms).to(device=device)
    lms[..., 1] = img_h - lms[..., 1] # flip the height axis

    if keypoint_mode == 'mediapipe':
        cal_lan_loss_fn = cal_lan_loss_mp
        out_name = img_name.replace("/images_512/", "/coeff_fit_mp/").replace(".png", "_coeff_fit_mp.npy")
    else:
        cal_lan_loss_fn = cal_lan_loss
        out_name = img_name.replace("/images_512/", "/coeff_fit_lm68/").replace(".png", "_coeff_fit_lm68.npy")
    try:
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
    except:
        pass

    id_dim, exp_dim = 80, 64
    sel_ids = np.arange(0, num_frames, 40)
    sel_num = sel_ids.shape[0]
    arg_focal = face_model.focal

    h = w = face_model.center * 2
    img_scale_factor = img_h / h
    lms /= img_scale_factor
    cxy = torch.tensor((w / 2.0, h / 2.0), dtype=torch.float).to(device=device)

    id_para = lms.new_zeros((num_frames, id_dim), requires_grad=True) # lms.new_zeros((1, id_dim), requires_grad=True)
    exp_para = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
    euler_angle = lms.new_zeros((num_frames, 3), requires_grad=True)
    trans = lms.new_zeros((num_frames, 3), requires_grad=True)

    focal_length = lms.new_zeros(1, requires_grad=True)
    focal_length.data += arg_focal

    set_requires_grad([id_para, exp_para, euler_angle, trans])

    optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
    optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=.1)

    # 其他参数初始化，先训练euler和trans
    for _ in range(200):
        proj_geo = face_model.compute_for_landmark_fit(
            id_para, exp_para, euler_angle, trans)
        loss_lan = cal_lan_loss_fn(proj_geo[:, :, :2], lms.detach())
        loss = loss_lan
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_frame.step()
    # print(f"loss_lan: {loss_lan.item():.2f}, euler_abs_mean: {euler_angle.abs().mean().item():.4f}, euler_std: {euler_angle.std().item():.4f}, euler_min: {euler_angle.min().item():.4f}, euler_max: {euler_angle.max().item():.4f}")
    # print(f"trans_z_mean: {trans[...,2].mean().item():.4f}, trans_z_std: {trans[...,2].std().item():.4f}, trans_min: {trans[...,2].min().item():.4f}, trans_max: {trans[...,2].max().item():.4f}")

    for param_group in optimizer_frame.param_groups:
        param_group['lr'] = 0.1

    # "jointly roughly training id exp euler trans"
    for _ in range(200):
        proj_geo = face_model.compute_for_landmark_fit(
            id_para, exp_para, euler_angle, trans)
        loss_lan = cal_lan_loss_fn(
            proj_geo[:, :, :2], lms.detach())
        loss_regid = torch.mean(id_para*id_para) # 正则化
        loss_regexp = torch.mean(exp_para * exp_para)

        loss = loss_lan  + loss_regid * LAMBDA_REG_ID + loss_regexp * LAMBDA_REG_EXP
        optimizer_idexp.zero_grad()
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_idexp.step()
        optimizer_frame.step()
    # print(f"loss_lan: {loss_lan.item():.2f}, loss_reg_id: {loss_regid.item():.2f},loss_reg_exp: {loss_regexp.item():.2f},")
    # print(f"euler_abs_mean: {euler_angle.abs().mean().item():.4f}, euler_std: {euler_angle.std().item():.4f}, euler_min: {euler_angle.min().item():.4f}, euler_max: {euler_angle.max().item():.4f}")
    # print(f"trans_z_mean: {trans[...,2].mean().item():.4f}, trans_z_std: {trans[...,2].std().item():.4f}, trans_min: {trans[...,2].min().item():.4f}, trans_max: {trans[...,2].max().item():.4f}")

    # start fine training, intialize from the roughly trained results
    id_para_ = lms.new_zeros((num_frames, id_dim), requires_grad=True)
    id_para_.data = id_para.data.clone()
    id_para = id_para_
    exp_para_ = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
    exp_para_.data = exp_para.data.clone()
    exp_para = exp_para_
    euler_angle_ = lms.new_zeros((num_frames, 3), requires_grad=True)
    euler_angle_.data = euler_angle.data.clone()
    euler_angle = euler_angle_
    trans_ = lms.new_zeros((num_frames, 3), requires_grad=True)
    trans_.data = trans.data.clone()
    trans = trans_

    batch_size = 1

    # "fine fitting the 3DMM in batches"
    for i in range(int((num_frames-1)/batch_size+1)):
        if (i+1)*batch_size > num_frames:
            start_n = num_frames-batch_size
            sel_ids = np.arange(max(num_frames-batch_size,0), num_frames)
        else:
            start_n = i*batch_size
            sel_ids = np.arange(i*batch_size, i*batch_size+batch_size)
        sel_lms = lms[sel_ids]

        sel_id_para = id_para.new_zeros(
            (batch_size, id_dim), requires_grad=True)
        sel_id_para.data = id_para[sel_ids].clone()
        sel_exp_para = exp_para.new_zeros(
            (batch_size, exp_dim), requires_grad=True)
        sel_exp_para.data = exp_para[sel_ids].clone()
        sel_euler_angle = euler_angle.new_zeros(
            (batch_size, 3), requires_grad=True)
        sel_euler_angle.data = euler_angle[sel_ids].clone()
        sel_trans = trans.new_zeros((batch_size, 3), requires_grad=True)
        sel_trans.data = trans[sel_ids].clone()
        
        set_requires_grad([sel_id_para, sel_exp_para, sel_euler_angle, sel_trans])
        optimizer_cur_batch = torch.optim.Adam(
            [sel_id_para, sel_exp_para, sel_euler_angle, sel_trans], lr=0.005)

        for j in range(50):
            proj_geo = face_model.compute_for_landmark_fit(
                sel_id_para, sel_exp_para, sel_euler_angle, sel_trans)
            loss_lan = cal_lan_loss_fn(
                proj_geo[:, :, :2], lms.unsqueeze(0).detach())

            loss_regid = torch.mean(sel_id_para*sel_id_para) # 正则化
            loss_regexp = torch.mean(sel_exp_para*sel_exp_para)
            loss = loss_lan + loss_regid * LAMBDA_REG_ID + loss_regexp * LAMBDA_REG_EXP 
            optimizer_cur_batch.zero_grad()
            loss.backward()
            optimizer_cur_batch.step()
        print(f"batch {i} | loss_lan: {loss_lan.item():.2f}, loss_reg_id: {loss_regid.item():.2f},loss_reg_exp: {loss_regexp.item():.2f}")
        id_para[sel_ids].data = sel_id_para.data.clone()
        exp_para[sel_ids].data = sel_exp_para.data.clone()
        euler_angle[sel_ids].data = sel_euler_angle.data.clone()
        trans[sel_ids].data = sel_trans.data.clone()

    coeff_dict = {'id': id_para.detach().cpu().numpy(), 'exp': exp_para.detach().cpu().numpy(),
                'euler': euler_angle.detach().cpu().numpy(), 'trans': trans.detach().cpu().numpy()}
    if save:
        np.save(out_name, coeff_dict, allow_pickle=True)
    
    if debug:
        import imageio
        debug_name = img_name.replace("/images_512/", "/coeff_fit_mp_debug/").replace(".png", "_debug.png").replace(".jpg", "_debug.jpg")
        try: os.makedirs(os.path.dirname(debug_name), exist_ok=True)
        except: pass
        proj_geo = face_model.compute_for_landmark_fit(id_para, exp_para, euler_angle, trans)
        lm68s = proj_geo[:,:,:2].detach().cpu().numpy()  # [T, 68,2]
        lm68s = lm68s * img_scale_factor
        lms = lms * img_scale_factor
        lm68s[..., 1] = img_h - lm68s[..., 1] # flip the height axis
        lms[..., 1] = img_h - lms[..., 1] # flip the height axis
        lm68s = lm68s.astype(int)
        lm68s = lm68s.reshape([-1,2])
        lms = lms.cpu().numpy().astype(int).reshape([-1,2])
        for lm in lm68s:
            img = cv2.circle(img, lm, 1, (0, 0, 255), thickness=-1)
        for gt_lm in lms:
            img = cv2.circle(img, gt_lm, 2, (255, 0, 0), thickness=1)
        imageio.imwrite(debug_name, img)
        print(f"debug img saved at {debug_name}")
    return coeff_dict

def out_exist_job(vid_name):
    out_name = vid_name.replace("/images_512/", "/coeff_fit_mp/").replace(".png","_coeff_fit_mp.npy") 
    # if os.path.exists(out_name) or not os.path.exists(lms_name):
    if os.path.exists(out_name):
        return None
    else:
        return vid_name

def get_todo_img_names(img_names):
    todo_img_names = []
    for i, res in multiprocess_run_tqdm(out_exist_job, img_names, num_workers=16):
        if res is not None:
            todo_img_names.append(res)
    return todo_img_names


if __name__ == '__main__':
    import argparse, glob, tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default='/home/tiger/datasets/raw/FFHQ/images_512')
    parser.add_argument("--ds_name", default='FFHQ')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--keypoint_mode", default='mediapipe', type=str)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--reset", action='store_true')
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--output_log", action='store_true')
    parser.add_argument("--load_names", action="store_true")

    args = parser.parse_args()
    img_dir = args.img_dir
    load_names = args.load_names
    
    print(f"args {args}")
    
    if args.ds_name == 'single_img':
        img_names = [img_dir]
    else:
        img_names_path = os.path.join(img_dir, "img_dir.pkl")
        if os.path.exists(img_names_path) and load_names:
            print(f"loading vid names from {img_names_path}")
            img_names = load_file(img_names_path)
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
                img_name_patterns = ["ref/*/*.png"]
                img_names = []
                for img_name_pattern in img_name_patterns:
                    img_name_pattern_full = os.path.join(img_dir, img_name_pattern)
                    img_names_part = glob.glob(img_name_pattern_full)
                    img_names.extend(img_names_part)
                img_names = sorted(img_names)
            print(f"saving image names to {img_names_path}")
            save_file(img_names_path, img_names)
            
    # import random
    # random.seed(args.seed)
    # random.shuffle(img_names)

    face_model = ParametricFaceModel(bfm_folder='deep_3drecon/BFM', 
                camera_distance=10, focal=1015, keypoint_mode=args.keypoint_mode)
    face_model.to(torch.device(args.device))
     
    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process -1 and process_id >= 0
        num_samples_per_process = len(img_names) // total_process
        if process_id == total_process:
            img_names = img_names[process_id * num_samples_per_process : ]
        else:
            img_names = img_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
    print(f"image names number (before fileter): {len(img_names)}")


    if not args.reset:
        img_names = get_todo_img_names(img_names)

    print(f"image names number (after  fileter): {len(img_names)}")
    for i in tqdm.trange(len(img_names), desc=f"process {process_id}: fitting 3dmm ..."):
        img_name = img_names[i]
        try:
            fit_3dmm_for_a_image(img_name, args.debug, device=args.device)
        except Exception as e:
            print(img_name, e)
        if args.output_log and i % max(int(len(img_names) * 0.003), 1) == 0:
            print(f"process {process_id}: {i + 1} / {len(img_names)} done")
            sys.stdout.flush()
            sys.stderr.flush()
            
    print(f"process {process_id}: fitting 3dmm all done")

