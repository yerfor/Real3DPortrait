import os
import sys
sys.path.append('./')
import torch
import torch.nn.functional as F
import torchshow as ts
import librosa
import random
import time
import numpy as np
import importlib
import tqdm
import copy
import cv2
import math

# common utils
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda, convert_to_tensor
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
# 3DMM-related utils
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_util.face3d_helper import Face3DHelper
from data_gen.utils.process_image.fit_3dmm_landmark import fit_3dmm_for_a_image
from data_gen.utils.process_video.fit_3dmm_landmark import fit_3dmm_for_a_video
from deep_3drecon.secc_renderer import SECC_Renderer
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
from data_gen.utils.process_image.extract_lm2d import extract_lms_mediapipe_job

# Face Parsing 
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background
# other inference utils
from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd
from inference.edit_secc import blink_eye_for_secc


def read_first_frame_from_a_video(vid_name):
    frames = []
    cap = cv2.VideoCapture(vid_name)
    ret, frame_bgr = cap.read()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb

def analyze_weights_img(gen_output):
    img_raw = gen_output['image_raw']
    mask_005_to_03 = torch.bitwise_and(gen_output['weights_img']>0.05, gen_output['weights_img']<0.3).repeat([1,3,1,1])
    mask_005_to_05 = torch.bitwise_and(gen_output['weights_img']>0.05, gen_output['weights_img']<0.5).repeat([1,3,1,1])
    mask_005_to_07 = torch.bitwise_and(gen_output['weights_img']>0.05, gen_output['weights_img']<0.7).repeat([1,3,1,1])
    mask_005_to_09 = torch.bitwise_and(gen_output['weights_img']>0.05, gen_output['weights_img']<0.9).repeat([1,3,1,1])
    mask_005_to_10 = torch.bitwise_and(gen_output['weights_img']>0.05, gen_output['weights_img']<1.0).repeat([1,3,1,1])

    img_raw_005_to_03 = img_raw.clone()
    img_raw_005_to_03[~mask_005_to_03] = -1
    img_raw_005_to_05 = img_raw.clone()
    img_raw_005_to_05[~mask_005_to_05] = -1
    img_raw_005_to_07 = img_raw.clone()
    img_raw_005_to_07[~mask_005_to_07] = -1
    img_raw_005_to_09 = img_raw.clone()
    img_raw_005_to_09[~mask_005_to_09] = -1
    img_raw_005_to_10 = img_raw.clone()
    img_raw_005_to_10[~mask_005_to_10] = -1
    ts.save([img_raw_005_to_03[0], img_raw_005_to_05[0], img_raw_005_to_07[0], img_raw_005_to_09[0], img_raw_005_to_10[0]])

def cal_face_area_percent(img_name):
    img = cv2.resize(cv2.imread(img_name)[:,:,::-1], (512,512))
    lm478 = extract_lms_mediapipe_job(img) / 512
    min_x = lm478[:,0].min()
    max_x = lm478[:,0].max()
    min_y = lm478[:,1].min()
    max_y = lm478[:,1].max()
    area = (max_x - min_x) * (max_y - min_y)
    return area

def crop_img_on_face_area_percent(img_name, out_name='temp/cropped_src_img.png', min_face_area_percent=0.2):
    try:
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
    except: pass
    face_area_percent = cal_face_area_percent(img_name)
    if face_area_percent >= min_face_area_percent:
        print(f"face area percent {face_area_percent} larger than threshold {min_face_area_percent}, directly use the input image...")
        cmd = f"cp {img_name} {out_name}"
        os.system(cmd)
        return out_name
    else:
        print(f"face area percent {face_area_percent} smaller than threshold {min_face_area_percent}, crop the input image...")
        img = cv2.resize(cv2.imread(img_name)[:,:,::-1], (512,512))
        lm478 = extract_lms_mediapipe_job(img).astype(int)
        min_x = lm478[:,0].min()
        max_x = lm478[:,0].max()
        min_y = lm478[:,1].min()
        max_y = lm478[:,1].max()
        face_area = (max_x - min_x) * (max_y - min_y)
        target_total_area = face_area / min_face_area_percent
        target_hw = int(target_total_area**0.5)
        center_x, center_y = (min_x+max_x)/2, (min_y+max_y)/2
        shrink_pixels = 2 * max(-(center_x - target_hw/2), center_x + target_hw/2 - 512, -(center_y - target_hw/2), center_y + target_hw/2-512)
        shrink_pixels = max(0, shrink_pixels)
        hw = math.floor(target_hw - shrink_pixels)
        new_min_x = int(center_x - hw/2)
        new_max_x = int(center_x + hw/2)
        new_min_y = int(center_y - hw/2)
        new_max_y = int(center_y + hw/2)

        img = img[new_min_y:new_max_y, new_min_x:new_max_x]
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(out_name, img[:,:,::-1])
        return out_name


class GeneFace2Infer:
    def __init__(self, audio2secc_dir, head_model_dir, torso_model_dir, device=None, inp=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.audio2secc_model = self.load_audio2secc(audio2secc_dir)
        self.secc2video_model = self.load_secc2video(head_model_dir, torso_model_dir, inp)
        self.audio2secc_model.to(device).eval()
        self.secc2video_model.to(device).eval()
        self.seg_model = MediapipeSegmenter()
        self.secc_renderer = SECC_Renderer(512)
        self.face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='lm68')
        self.mp_face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='mediapipe')

    def load_audio2secc(self, audio2secc_dir):
        config_name = f"{audio2secc_dir}/config.yaml" if not audio2secc_dir.endswith(".ckpt") else f"{os.path.dirname(audio2secc_dir)}/config.yaml"
        set_hparams(f"{config_name}", print_hparams=False)
        self.audio2secc_dir = audio2secc_dir
        self.audio2secc_hparams = copy.deepcopy(hparams)
        from modules.audio2motion.vae import VAEModel, PitchContourVAEModel
        if self.audio2secc_hparams['audio_type'] == 'hubert':
            audio_in_dim = 1024
        elif self.audio2secc_hparams['audio_type'] == 'mfcc':
            audio_in_dim = 13

        if 'icl' in hparams['task_cls']:
            self.use_icl_audio2motion = True
            model = InContextAudio2MotionModel(hparams['icl_model_type'], hparams=self.audio2secc_hparams)
        else:
            self.use_icl_audio2motion = False
            if hparams.get("use_pitch", False) is True:
                model = PitchContourVAEModel(hparams, in_out_dim=64, audio_in_dim=audio_in_dim)
            else:
                model = VAEModel(in_out_dim=64, audio_in_dim=audio_in_dim)
        load_ckpt(model, f"{audio2secc_dir}", model_name='model', strict=True)
        return model

    def load_secc2video(self, head_model_dir, torso_model_dir, inp):
        if inp is None:
            inp = {}
        self.head_model_dir = head_model_dir
        self.torso_model_dir = torso_model_dir
        if torso_model_dir != '':
            if torso_model_dir.endswith(".ckpt"):
                set_hparams(f"{os.path.dirname(torso_model_dir)}/config.yaml", print_hparams=False)
            else:
                set_hparams(f"{torso_model_dir}/config.yaml", print_hparams=False)
            if inp.get('head_torso_threshold', None) is not None:
                hparams['htbsr_head_threshold'] = inp['head_torso_threshold']
            self.secc2video_hparams = copy.deepcopy(hparams)
            from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane_Torso
            model = OSAvatarSECC_Img2plane_Torso()
            load_ckpt(model, f"{torso_model_dir}", model_name='model', strict=True)
            if head_model_dir != '':
                print("| Warning: Assigned --torso_ckpt which also contains head, but --head_ckpt is also assigned, skipping the --head_ckpt.")
        else:
            from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane
            if head_model_dir.endswith(".ckpt"):
                set_hparams(f"{os.path.dirname(head_model_dir)}/config.yaml", print_hparams=False)
            else:
                set_hparams(f"{head_model_dir}/config.yaml", print_hparams=False)
            if inp.get('head_torso_threshold', None) is not None:
                hparams['htbsr_head_threshold'] = inp['head_torso_threshold']
            self.secc2video_hparams = copy.deepcopy(hparams)
            model = OSAvatarSECC_Img2plane()
            load_ckpt(model, f"{head_model_dir}", model_name='model', strict=True)
        return model

    def infer_once(self, inp):
        self.inp = inp
        samples = self.prepare_batch_from_inp(inp)
        seed = inp['seed'] if inp['seed'] is not None else int(time.time())
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        out_name = self.forward_system(samples, inp)
        return out_name
    
    def prepare_batch_from_inp(self, inp):
        """
        :param inp: {'audio_source_name': (str)}
        :return: a dict that contains the condition feature of NeRF
        """
        tmp_img_name = 'infer_out/tmp/cropped_src_img.png'
        crop_img_on_face_area_percent(inp['src_image_name'], tmp_img_name, min_face_area_percent=inp['min_face_area_percent'])
        inp['src_image_name'] = tmp_img_name

        sample = {}
        # Process Driving Motion
        if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
            self.save_wav16k(inp['drv_audio_name'])
            if self.audio2secc_hparams['audio_type'] == 'hubert':
                hubert = self.get_hubert(self.wav16k_name)
            elif self.audio2secc_hparams['audio_type'] == 'mfcc':
                hubert = self.get_mfcc(self.wav16k_name) / 100

            f0 = self.get_f0(self.wav16k_name)
            if f0.shape[0] > len(hubert):
                f0 = f0[:len(hubert)]
            else:
                num_to_pad = len(hubert) - len(f0)
                f0 = np.pad(f0, pad_width=((0,num_to_pad), (0,0)))
            t_x = hubert.shape[0]
            x_mask = torch.ones([1, t_x]).float() # mask for audio frames
            y_mask = torch.ones([1, t_x//2]).float() # mask for motion/image frames
            sample.update({
                'hubert': torch.from_numpy(hubert).float().unsqueeze(0).cuda(),
                'f0': torch.from_numpy(f0).float().reshape([1,-1]).cuda(),
                'x_mask': x_mask.cuda(),
                'y_mask': y_mask.cuda(),
                })
            sample['blink'] = torch.zeros([1, t_x, 1]).long().cuda()
            sample['audio'] = sample['hubert']
            sample['eye_amp'] = torch.ones([1, 1]).cuda() * 1.0
            sample['mouth_amp'] = torch.ones([1, 1]).cuda() * inp['mouth_amp']
        elif inp['drv_audio_name'][-4:] in ['.mp4']:
            drv_motion_coeff_dict = fit_3dmm_for_a_video(inp['drv_audio_name'], save=False)
            drv_motion_coeff_dict = convert_to_tensor(drv_motion_coeff_dict)
            t_x = drv_motion_coeff_dict['exp'].shape[0] * 2
            self.drv_motion_coeff_dict = drv_motion_coeff_dict
        elif inp['drv_audio_name'][-4:] in ['.npy']:
            drv_motion_coeff_dict = np.load(inp['drv_audio_name'], allow_pickle=True).tolist()
            drv_motion_coeff_dict = convert_to_tensor(drv_motion_coeff_dict)
            t_x = drv_motion_coeff_dict['exp'].shape[0] * 2
            self.drv_motion_coeff_dict = drv_motion_coeff_dict

        # Face Parsing
        image_name = inp['src_image_name']
        if image_name.endswith(".mp4"):
            img = read_first_frame_from_a_video(image_name)
            image_name = inp['src_image_name'] = image_name[:-4] + '.png'
            cv2.imwrite(image_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sample['ref_gt_img'] = load_img_to_normalized_512_bchw_tensor(image_name).cuda()
        img = load_img_to_512_hwc_array(image_name)
        segmap = self.seg_model._cal_seg_map(img)
        sample['segmap'] = torch.tensor(segmap).float().unsqueeze(0).cuda()
        head_img = self.seg_model._seg_out_img_with_segmap(img, segmap, mode='head')[0]
        sample['ref_head_img'] = ((torch.tensor(head_img) - 127.5)/127.5).float().unsqueeze(0).permute(0, 3, 1,2).cuda() # [b,c,h,w]
        ts.save(sample['ref_head_img'])
        inpaint_torso_img, _, _, _ = inpaint_torso_job(img, segmap)
        sample['ref_torso_img'] = ((torch.tensor(inpaint_torso_img) - 127.5)/127.5).float().unsqueeze(0).permute(0, 3, 1,2).cuda() # [b,c,h,w]
        
        if inp['bg_image_name'] == '':
            bg_img = extract_background([img], [segmap], 'knn')
        else:
            bg_img = cv2.imread(inp['bg_image_name'])
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (512,512))
        sample['bg_img'] = ((torch.tensor(bg_img) - 127.5)/127.5).float().unsqueeze(0).permute(0, 3, 1,2).cuda() # [b,c,h,w]

        # 3DMM, get identity code and camera pose
        coeff_dict = fit_3dmm_for_a_image(image_name, save=False)
        assert coeff_dict is not None
        src_id = torch.tensor(coeff_dict['id']).reshape([1,80]).cuda()
        src_exp = torch.tensor(coeff_dict['exp']).reshape([1,64]).cuda()
        src_euler = torch.tensor(coeff_dict['euler']).reshape([1,3]).cuda()
        src_trans = torch.tensor(coeff_dict['trans']).reshape([1,3]).cuda()
        sample['id'] = src_id.repeat([t_x//2,1])

        # get the src_kp for torso model
        src_kp = self.face3d_helper.reconstruct_lm2d(src_id, src_exp, src_euler, src_trans) # [1, 68, 2]
        src_kp = (src_kp-0.5) / 0.5 # rescale to -1~1
        sample['src_kp'] = torch.clamp(src_kp, -1, 1).repeat([t_x//2,1,1])

        # get camera pose file
        # random.seed(time.time())
        inp['drv_pose_name'] = inp['drv_pose_name']
        print(f"| To extract pose from {inp['drv_pose_name']}")

        # extract camera pose 
        if inp['drv_pose_name'] == 'static':
            sample['euler'] = torch.tensor(coeff_dict['euler']).reshape([1,3]).cuda().repeat([t_x//2,1]) # default static pose
            sample['trans'] = torch.tensor(coeff_dict['trans']).reshape([1,3]).cuda().repeat([t_x//2,1])
        else: # from file
            if inp['drv_pose_name'].endswith('.mp4'):
                # extract coeff from video
                drv_pose_coeff_dict = fit_3dmm_for_a_video(inp['drv_pose_name'], save=False)
            else:
                # load from npy
                drv_pose_coeff_dict = np.load(inp['drv_pose_name'], allow_pickle=True).tolist()
            print(f"| Extracted pose from {inp['drv_pose_name']}")
            eulers = convert_to_tensor(drv_pose_coeff_dict['euler']).reshape([-1,3]).cuda()
            trans = convert_to_tensor(drv_pose_coeff_dict['trans']).reshape([-1,3]).cuda()
            len_pose = len(eulers)
            index_lst = [mirror_index(i, len_pose) for i in range(t_x//2)]
            sample['euler'] = eulers[index_lst]
            sample['trans'] = trans[index_lst]

        # fix the z axis
        sample['trans'][:, -1] = sample['trans'][0:1, -1].repeat([sample['trans'].shape[0]])

        # mapping to the init pose
        print(inp)
        if inp.get("map_to_init_pose", 'True') in ['True', True]:
            diff_euler = torch.tensor(coeff_dict['euler']).reshape([1,3]).cuda() - sample['euler'][0:1]
            sample['euler'] = sample['euler'] + diff_euler
            diff_trans = torch.tensor(coeff_dict['trans']).reshape([1,3]).cuda() - sample['trans'][0:1]
            sample['trans'] = sample['trans'] + diff_trans

        # prepare camera
        camera_ret = get_eg3d_convention_camera_pose_intrinsic({'euler':sample['euler'].cpu(), 'trans':sample['trans'].cpu()})
        c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
        # smooth camera
        camera_smo_ksize = 7
        camera = np.concatenate([c2w.reshape([-1,16]), intrinsics.reshape([-1,9])], axis=-1)
        camera = smooth_camera_sequence(camera, kernel_size=camera_smo_ksize) # [T, 25]
        camera = torch.tensor(camera).cuda().float()
        sample['camera'] = camera

        return sample

    @torch.no_grad()
    def get_hubert(self, wav16k_name):
        from data_gen.utils.process_audio.extract_hubert import get_hubert_from_16k_wav
        hubert = get_hubert_from_16k_wav(wav16k_name).detach().numpy()
        len_mel = hubert.shape[0]
        x_multiply = 8
        if len_mel % x_multiply == 0:
            num_to_pad = 0
        else:
            num_to_pad = x_multiply - len_mel % x_multiply
        hubert = np.pad(hubert, pad_width=((0,num_to_pad), (0,0)))
        return hubert

    def get_mfcc(self, wav16k_name):
        from utils.audio import librosa_wav2mfcc
        hparams['fft_size'] = 1200
        hparams['win_size'] = 1200
        hparams['hop_size'] = 480
        hparams['audio_num_mel_bins'] = 80
        hparams['fmin'] = 80
        hparams['fmax'] = 12000
        hparams['audio_sample_rate'] = 24000
        mfcc = librosa_wav2mfcc(wav16k_name,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            center=True)
        mfcc = np.array(mfcc).reshape([-1, 13])
        len_mel = mfcc.shape[0]
        x_multiply = 8
        if len_mel % x_multiply == 0:
            num_to_pad = 0
        else:
            num_to_pad = x_multiply - len_mel % x_multiply
        mfcc = np.pad(mfcc, pad_width=((0,num_to_pad), (0,0)))
        return mfcc

    @torch.no_grad()
    def forward_audio2secc(self, batch, inp=None):
        if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
            # audio-to-exp
            ret = {}
            pred = self.audio2secc_model.forward(batch, ret=ret,train=False, temperature=inp['temperature'],)
            print("| audio-to-motion finished")
            if pred.shape[-1] == 144:
                id = ret['pred'][0][:,:80]
                exp = ret['pred'][0][:,80:]
            else:
                id = batch['id']
                exp = ret['pred'][0]
            if len(id) < len(exp): # happens when use ICL
                id = torch.cat([id, id[0].unsqueeze(0).repeat([len(exp)-len(id),1])])
            batch['id'] = id
            batch['exp'] = exp
        else:
            drv_motion_coeff_dict = self.drv_motion_coeff_dict
            batch['exp'] = torch.FloatTensor(drv_motion_coeff_dict['exp']).cuda()

        batch = self.get_driving_motion(batch['id'], batch['exp'], batch['euler'], batch['trans'], batch, inp)
        if self.use_icl_audio2motion:
            self.audio2secc_model.empty_context()
        return batch

    @torch.no_grad()
    def get_driving_motion(self, id, exp, euler, trans, batch, inp):
        zero_eulers = torch.zeros([id.shape[0], 3]).to(id.device)
        zero_trans = torch.zeros([id.shape[0], 3]).to(exp.device)
        # render the secc given the id,exp
        with torch.no_grad():
            chunk_size = 50
            drv_secc_color_lst = []
            num_iters = len(id)//chunk_size if len(id)%chunk_size == 0 else len(id)//chunk_size+1
            for i in tqdm.trange(num_iters, desc="rendering drv secc"):
                torch.cuda.empty_cache()
                face_mask, drv_secc_color = self.secc_renderer(id[i*chunk_size:(i+1)*chunk_size], exp[i*chunk_size:(i+1)*chunk_size], zero_eulers[i*chunk_size:(i+1)*chunk_size], zero_trans[i*chunk_size:(i+1)*chunk_size])
                drv_secc_color_lst.append(drv_secc_color.cpu())
        drv_secc_colors = torch.cat(drv_secc_color_lst, dim=0)
        _, src_secc_color = self.secc_renderer(id[0:1], exp[0:1], zero_eulers[0:1], zero_trans[0:1])
        _, cano_secc_color = self.secc_renderer(id[0:1], exp[0:1]*0, zero_eulers[0:1], zero_trans[0:1])
        batch['drv_secc'] = drv_secc_colors.cuda()
        batch['src_secc'] = src_secc_color.cuda()
        batch['cano_secc'] = cano_secc_color.cuda()
        
        # blinking secc
        if inp['blink_mode'] == 'period':
            period = 5 # second

            for i in tqdm.trange(len(drv_secc_colors),desc="blinking secc"):
                if i % (25*period) == 0:
                    blink_dur_frames = random.randint(8, 12)
                    for offset in range(blink_dur_frames):
                        j = offset + i
                        if j >= len(drv_secc_colors)-1: break
                        def blink_percent_fn(t, T):
                            return -4/T**2 * t**2 + 4/T * t
                        blink_percent = blink_percent_fn(offset, blink_dur_frames)
                        secc = batch['drv_secc'][j]
                        out_secc = blink_eye_for_secc(secc, blink_percent)
                        out_secc = out_secc.cuda()
                        batch['drv_secc'][j] = out_secc

        # get the drv_kp for torso model, using the transformed trajectory
        drv_kp = self.face3d_helper.reconstruct_lm2d(id, exp, euler, trans) # [T, 68, 2]

        drv_kp = (drv_kp-0.5) / 0.5 # rescale to -1~1
        batch['drv_kp'] = torch.clamp(drv_kp, -1, 1)
        return batch

    @torch.no_grad()
    def forward_secc2video(self, batch, inp=None):
        num_frames = len(batch['drv_secc'])
        camera = batch['camera']
        src_kps = batch['src_kp']
        drv_kps = batch['drv_kp']
        cano_secc_color = batch['cano_secc']
        src_secc_color = batch['src_secc']
        drv_secc_colors = batch['drv_secc']
        ref_img_gt = batch['ref_gt_img']
        ref_img_head = batch['ref_head_img']
        ref_torso_img = batch['ref_torso_img']
        bg_img = batch['bg_img']
        segmap = batch['segmap']
        
        # smooth torso drv_kp
        torso_smo_ksize = 7
        drv_kps = smooth_features_xd(drv_kps.reshape([-1, 68*2]), kernel_size=torso_smo_ksize).reshape([-1, 68, 2])

        # forward renderer
        if inp['low_memory_usage']:
            # save memory, when one image is rendered, write it into video
            import imageio
            debug_name = 'demo.mp4'
            writer = imageio.get_writer(debug_name, fps=25, format='FFMPEG', codec='h264')
            
            with torch.no_grad():
                for i in tqdm.trange(num_frames, desc="Real3D-Portrait is rendering frames"):
                    kp_src = torch.cat([src_kps[i:i+1].reshape([1, 68, 2]), torch.zeros([1, 68,1]).to(src_kps.device)],dim=-1)
                    kp_drv = torch.cat([drv_kps[i:i+1].reshape([1, 68, 2]), torch.zeros([1, 68,1]).to(drv_kps.device)],dim=-1)
                    cond={'cond_cano': cano_secc_color,'cond_src': src_secc_color, 'cond_tgt': drv_secc_colors[i:i+1].cuda(),
                            'ref_torso_img': ref_torso_img, 'bg_img': bg_img, 'segmap': segmap,
                            'kp_s': kp_src, 'kp_d': kp_drv}
                    if i == 0:
                        gen_output = self.secc2video_model.forward(img=ref_img_head, camera=camera[i:i+1], cond=cond, ret={}, cache_backbone=True, use_cached_backbone=False)
                    else:
                        gen_output = self.secc2video_model.forward(img=ref_img_head, camera=camera[i:i+1], cond=cond, ret={}, cache_backbone=False, use_cached_backbone=True)
                    img = ((gen_output['image']+1)/2 * 255.).permute(0, 2, 3, 1)[0].int().cpu().numpy().astype(np.uint8)
                    writer.append_data(img)
            writer.close()
        else:
            img_raw_lst = []
            img_lst = []
            depth_img_lst = []
            with torch.no_grad():
                for i in tqdm.trange(num_frames, desc="Real3D-Portrait is rendering frames"):
                    kp_src = torch.cat([src_kps[i:i+1].reshape([1, 68, 2]), torch.zeros([1, 68,1]).to(src_kps.device)],dim=-1)
                    kp_drv = torch.cat([drv_kps[i:i+1].reshape([1, 68, 2]), torch.zeros([1, 68,1]).to(drv_kps.device)],dim=-1)
                    cond={'cond_cano': cano_secc_color,'cond_src': src_secc_color, 'cond_tgt': drv_secc_colors[i:i+1].cuda(),
                            'ref_torso_img': ref_torso_img, 'bg_img': bg_img, 'segmap': segmap,
                            'kp_s': kp_src, 'kp_d': kp_drv}
                    if i == 0:
                        gen_output = self.secc2video_model.forward(img=ref_img_head, camera=camera[i:i+1], cond=cond, ret={}, cache_backbone=True, use_cached_backbone=False)
                    else:
                        gen_output = self.secc2video_model.forward(img=ref_img_head, camera=camera[i:i+1], cond=cond, ret={}, cache_backbone=False, use_cached_backbone=True)
                    img_lst.append(gen_output['image'])
                    img_raw_lst.append(gen_output['image_raw'])
                    depth_img_lst.append(gen_output['image_depth'])

            # save demo video
            depth_imgs = torch.cat(depth_img_lst)
            imgs = torch.cat(img_lst)
            imgs_raw = torch.cat(img_raw_lst)
            secc_img = torch.cat([torch.nn.functional.interpolate(drv_secc_colors[i:i+1], (512,512)) for i in range(num_frames)])
            
            if inp['out_mode'] == 'concat_debug':
                secc_img = secc_img.cpu()
                secc_img = ((secc_img + 1) * 127.5).permute(0, 2, 3, 1).int().numpy()

                depth_img = F.interpolate(depth_imgs, (512,512)).cpu()
                depth_img = depth_img.repeat([1,3,1,1])
                depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
                depth_img = depth_img * 2 - 1
                depth_img = depth_img.clamp(-1,1)

                secc_img = secc_img / 127.5 - 1
                secc_img = torch.from_numpy(secc_img).permute(0, 3, 1, 2)
                imgs = torch.cat([ref_img_gt.repeat([imgs.shape[0],1,1,1]).cpu(), secc_img, F.interpolate(imgs_raw, (512,512)).cpu(), depth_img, imgs.cpu()], dim=-1)
            elif inp['out_mode'] == 'final':
                imgs = imgs.cpu()
            elif inp['out_mode'] == 'debug':
                raise NotImplementedError("to do: save separate videos")
            imgs = imgs.clamp(-1,1)

            import imageio
            debug_name = 'demo.mp4'
            out_imgs = ((imgs.permute(0, 2, 3, 1) + 1)/2 * 255).int().cpu().numpy().astype(np.uint8)
            writer = imageio.get_writer(debug_name, fps=25, format='FFMPEG', codec='h264')
            
            for i in tqdm.trange(len(out_imgs), desc="Imageio is saving video"):
                writer.append_data(out_imgs[i])
            writer.close()
        
        # add audio track
        out_fname = 'infer_out/tmp/' + os.path.basename(inp['src_image_name'])[:-4] + '_' + os.path.basename(inp['drv_pose_name'])[:-4] + '.mp4' if inp['out_name'] == '' else inp['out_name']
        try:
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        except: pass
        if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
            os.system(f"ffmpeg -i {debug_name} -i {self.wav16k_name} -y -v quiet -shortest {out_fname}")
            os.system(f"rm {debug_name}")
            os.system(f"rm {self.wav16k_name}")
        else:
            ret = os.system(f"ffmpeg -i {debug_name} -i {inp['drv_audio_name']} -map 0:v -map 1:a -y -v quiet -shortest {out_fname}")
            if ret != 0: # 没有成功从drv_audio_name里面提取到音频, 则直接输出无音频轨道的纯视频
                os.system(f"mv {debug_name} {out_fname}")
        print(f"Saved at {out_fname}")
        return out_fname
        
    @torch.no_grad()
    def forward_system(self, batch, inp):
        self.forward_audio2secc(batch, inp)
        out_fname = self.forward_secc2video(batch, inp)
        return out_fname

    @classmethod
    def example_run(cls, inp=None):
        inp_tmp = {
            'drv_audio_name': 'data/raw/val_wavs/zozo.wav',
            'src_image_name': 'data/raw/val_imgs/Macron.png'
            }
        if inp is not None:
            inp_tmp.update(inp)
        inp = inp_tmp

        infer_instance = cls(inp['a2m_ckpt'], inp['head_ckpt'], inp['torso_ckpt'], inp=inp)
        infer_instance.infer_once(inp)

    ##############
    # IO-related
    ##############
    def save_wav16k(self, audio_name):
        supported_types = ('.wav', '.mp3', '.mp4', '.avi')
        assert audio_name.endswith(supported_types), f"Now we only support {','.join(supported_types)} as audio source!"
        wav16k_name = audio_name[:-4] + '_16k.wav'
        self.wav16k_name = wav16k_name
        extract_wav_cmd = f"ffmpeg -i {audio_name} -f wav -ar 16000 -v quiet -y {wav16k_name} -y"
        os.system(extract_wav_cmd)
        print(f"Extracted wav file (16khz) from {audio_name} to {wav16k_name}.")

    def get_f0(self, wav16k_name):
        from data_gen.utils.process_audio.extract_mel_f0 import extract_mel_from_fname, extract_f0_from_wav_and_mel
        wav, mel = extract_mel_from_fname(self.wav16k_name)
        f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
        f0 = f0.reshape([-1,1])
        return f0

if __name__ == '__main__':
    import argparse, glob, tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", default='checkpoints/240210_real3dportrait_orig/audio2secc_vae', type=str) 
    parser.add_argument("--head_ckpt", default='', type=str)
    parser.add_argument("--torso_ckpt", default='checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig', type=str) 
    parser.add_argument("--src_img", default='data/raw/examples/Macron.png', type=str) # data/raw/examples/Macron.png
    parser.add_argument("--bg_img", default='', type=str) # data/raw/examples/bg.png
    parser.add_argument("--drv_aud", default='data/raw/examples/Obama_5s.wav', type=str) # data/raw/examples/Obama_5s.wav
    parser.add_argument("--drv_pose", default='data/raw/examples/May_5s.mp4', type=str) # data/raw/examples/May_5s.mp4
    parser.add_argument("--blink_mode", default='period', type=str) # none | period
    parser.add_argument("--temperature", default=0.2, type=float) # sampling temperature in audio2motion, higher -> more diverse, less accurate
    parser.add_argument("--mouth_amp", default=0.45, type=float) # scale of predicted mouth, enabled in audio-driven
    parser.add_argument("--head_torso_threshold", default=None, type=float, help="0.1~1.0, turn up this value if the hair is translucent")
    parser.add_argument("--out_name", default='') # output filename
    parser.add_argument("--out_mode", default='concat_debug') # final: only output talking head video; concat_debug: talking head with internel features  
    parser.add_argument("--map_to_init_pose", default='True') # whether to map the pose of first frame to source image
    parser.add_argument("--seed", default=None, type=int) # random seed, default None to use time.time()
    parser.add_argument("--min_face_area_percent", default=0.2, type=float) # scale of predicted mouth, enabled in audio-driven
    parser.add_argument("--low_memory_usage", action='store_true', help='write img to video upon generated, leads to slower fps, but use less memory')

    args = parser.parse_args()

    inp = {
            'a2m_ckpt': args.a2m_ckpt,
            'head_ckpt': args.head_ckpt,
            'torso_ckpt': args.torso_ckpt,
            'src_image_name': args.src_img,
            'bg_image_name': args.bg_img,
            'drv_audio_name': args.drv_aud,
            'drv_pose_name': args.drv_pose,
            'blink_mode': args.blink_mode,
            'temperature': args.temperature,
            'mouth_amp': args.mouth_amp,
            'out_name': args.out_name,
            'out_mode': args.out_mode,
            'map_to_init_pose': args.map_to_init_pose,
            'head_torso_threshold': args.head_torso_threshold,
            'seed': args.seed,
            'min_face_area_percent': args.min_face_area_percent,
            'low_memory_usage': args.low_memory_usage,
            }

    GeneFace2Infer.example_run(inp)