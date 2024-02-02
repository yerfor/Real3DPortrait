import cv2
import torch
from utils.commons.image_utils import dilate, erode
from sklearn.neighbors import NearestNeighbors
import copy
import numpy as np
from utils.commons.meters import Timer

def hold_eye_opened_for_secc(img):
    img = img.permute(1,2,0).cpu().numpy()
    img = ((img +1)/2*255).astype(np.uint)
    face_mask = (img[...,0] != 0) & (img[...,1] != 0) & (img[...,2] != 0)
    face_xys = np.stack(np.nonzero(face_mask)).transpose(1, 0) # [N_nonbg,2] coordinate of non-face pixels
    h,w = face_mask.shape
    # get face and eye mask
    left_eye_prior_reigon = np.zeros([h,w], dtype=bool)
    right_eye_prior_reigon = np.zeros([h,w], dtype=bool)
    left_eye_prior_reigon[h//4:h//2, w//4:w//2] = True
    right_eye_prior_reigon[h//4:h//2, w//2:w//4*3] = True
    eye_prior_reigon = left_eye_prior_reigon | right_eye_prior_reigon
    coarse_eye_mask = (~ face_mask) & eye_prior_reigon
    coarse_eye_xys = np.stack(np.nonzero(coarse_eye_mask)).transpose(1, 0) # [N_nonbg,2] coordinate of non-face pixels

    opened_eye_mask = cv2.imread('inference/os_avatar/opened_eye_mask.png')
    opened_eye_mask = torch.nn.functional.interpolate(torch.tensor(opened_eye_mask).permute(2,0,1).unsqueeze(0), size=(img.shape[0], img.shape[1]), mode='nearest')[0].permute(1,2,0).sum(-1).bool().cpu() # [512,512,3]
    coarse_opened_eye_xys = np.stack(np.nonzero(opened_eye_mask)) # [N_nonbg,2] coordinate of non-face pixels
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coarse_eye_xys)
    dists, _ = nbrs.kneighbors(coarse_opened_eye_xys) # [512*512, 1] distance to nearest non-bg pixel
    # print(dists.max())
    non_opened_eye_pixs = dists > max(dists.max()*0.75, 4) # 大于这个距离的opened eye部分会被合上
    non_opened_eye_pixs = non_opened_eye_pixs.reshape([-1])
    opened_eye_xys_to_erode = coarse_opened_eye_xys[non_opened_eye_pixs]
    opened_eye_mask[opened_eye_xys_to_erode[...,0], opened_eye_xys_to_erode[...,1]] = False # shrink 将mask在face-eye边界收缩3pixel，为了平滑

    img[opened_eye_mask] = 0
    return torch.tensor(img.astype(np.float32) / 127.5 - 1).permute(2,0,1)
    

# def hold_eye_opened_for_secc(img):
#     img = copy.copy(img)
#     eye_mask = cv2.imread('inference/os_avatar/opened_eye_mask.png')
#     eye_mask = torch.nn.functional.interpolate(torch.tensor(eye_mask).permute(2,0,1).unsqueeze(0), size=(img.shape[-2], img.shape[-1]), mode='nearest')[0].bool().to(img.device) # [3,512,512]
#     img[eye_mask] = -1
#     return img
    
def blink_eye_for_secc(img, close_eye_percent=0.5):
    """
    secc_img: [3,h,w], tensor, -1~1
    """
    img = img.permute(1,2,0).cpu().numpy()
    img = ((img +1)/2*255).astype(np.uint)
    assert close_eye_percent <= 1.0 and close_eye_percent >= 0.
    if close_eye_percent == 0: return torch.tensor(img.astype(np.float32) / 127.5 - 1).permute(2,0,1)
    img = copy.deepcopy(img)
    face_mask = (img[...,0] != 0) & (img[...,1] != 0) & (img[...,2] != 0)
    h,w = face_mask.shape

    # get face and eye mask
    left_eye_prior_reigon = np.zeros([h,w], dtype=bool)
    right_eye_prior_reigon = np.zeros([h,w], dtype=bool)
    left_eye_prior_reigon[h//4:h//2, w//4:w//2] = True
    right_eye_prior_reigon[h//4:h//2, w//2:w//4*3] = True
    eye_prior_reigon = left_eye_prior_reigon | right_eye_prior_reigon
    coarse_eye_mask = (~ face_mask) & eye_prior_reigon
    coarse_left_eye_mask = (~ face_mask) & left_eye_prior_reigon
    coarse_right_eye_mask = (~ face_mask) & right_eye_prior_reigon
    coarse_eye_xys = np.stack(np.nonzero(coarse_eye_mask)).transpose(1, 0) # [N_nonbg,2] coordinate of non-face pixels
    min_h = coarse_eye_xys[:, 0].min()
    max_h = coarse_eye_xys[:, 0].max()
    coarse_left_eye_xys = np.stack(np.nonzero(coarse_left_eye_mask)).transpose(1, 0) # [N_nonbg,2] coordinate of non-face pixels
    left_min_w = coarse_left_eye_xys[:, 1].min()
    left_max_w = coarse_left_eye_xys[:, 1].max()
    coarse_right_eye_xys = np.stack(np.nonzero(coarse_right_eye_mask)).transpose(1, 0) # [N_nonbg,2] coordinate of non-face pixels
    right_min_w = coarse_right_eye_xys[:, 1].min()
    right_max_w = coarse_right_eye_xys[:, 1].max()

    # 尽力较少需要考虑的face_xyz,以降低KNN的损耗
    left_eye_prior_reigon = np.zeros([h,w], dtype=bool)
    more_room = 4 # 过小会导致一些问题
    left_eye_prior_reigon[min_h-more_room:max_h+more_room, left_min_w-more_room:left_max_w+more_room] = True
    right_eye_prior_reigon = np.zeros([h,w], dtype=bool)
    right_eye_prior_reigon[min_h-more_room:max_h+more_room, right_min_w-more_room:right_max_w+more_room] = True
    eye_prior_reigon = left_eye_prior_reigon | right_eye_prior_reigon

    around_eye_face_mask = face_mask & eye_prior_reigon
    face_mask = around_eye_face_mask
    face_xys = np.stack(np.nonzero(around_eye_face_mask)).transpose(1, 0) # [N_nonbg,2] coordinate of non-face pixels

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coarse_eye_xys)
    dists, _ = nbrs.kneighbors(face_xys) # [512*512, 1] distance to nearest non-bg pixel
    face_pixs = dists > 5 # 只有距离最近的eye pixel大于5的才被认为是face，过小会导致一些问题
    face_pixs = face_pixs.reshape([-1])
    face_xys_to_erode = face_xys[~face_pixs]
    face_mask[face_xys_to_erode[...,0], face_xys_to_erode[...,1]] = False # shrink 将mask在face-eye边界收缩3pixel，为了平滑
    eye_mask = (~ face_mask) & eye_prior_reigon

    h_grid = np.mgrid[0:h, 0:w][0]
    eye_num_pixel_along_w_axis = eye_mask.sum(axis=0)
    eye_mask_along_w_axis = eye_num_pixel_along_w_axis != 0

    tmp_h_grid = h_grid.copy()
    tmp_h_grid[~eye_mask] = 0
    eye_mean_h_coord_along_w_axis = tmp_h_grid.sum(axis=0) / np.clip(eye_num_pixel_along_w_axis, a_min=1, a_max=h)
    tmp_h_grid = h_grid.copy()
    tmp_h_grid[~eye_mask] = 99999
    eye_min_h_coord_along_w_axis = tmp_h_grid.min(axis=0)
    tmp_h_grid = h_grid.copy()
    tmp_h_grid[~eye_mask] = -99999
    eye_max_h_coord_along_w_axis = tmp_h_grid.max(axis=0)

    eye_low_h_coord_along_w_axis = close_eye_percent * eye_mean_h_coord_along_w_axis + (1-close_eye_percent) * eye_min_h_coord_along_w_axis # upper eye 
    eye_high_h_coord_along_w_axis = close_eye_percent * eye_mean_h_coord_along_w_axis + (1-close_eye_percent) * eye_max_h_coord_along_w_axis # lower eye 

    tmp_h_grid = h_grid.copy()
    tmp_h_grid[~eye_mask] = 99999
    upper_eye_blink_mask = tmp_h_grid <= eye_low_h_coord_along_w_axis
    tmp_h_grid = h_grid.copy()
    tmp_h_grid[~eye_mask] = -99999
    lower_eye_blink_mask = tmp_h_grid >= eye_high_h_coord_along_w_axis
    eye_blink_mask = upper_eye_blink_mask | lower_eye_blink_mask

    face_xys = np.stack(np.nonzero(around_eye_face_mask)).transpose(1, 0) # [N_nonbg,2] coordinate of non-face pixels
    eye_blink_xys = np.stack(np.nonzero(eye_blink_mask)).transpose(1, 0) # [N_nonbg,hw] coordinate of non-face pixels
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(face_xys)
    distances, indices = nbrs.kneighbors(eye_blink_xys)
    bg_fg_xys = face_xys[indices[:, 0]]
    img[eye_blink_xys[:, 0], eye_blink_xys[:, 1], :] = img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
    return torch.tensor(img.astype(np.float32) / 127.5 - 1).permute(2,0,1)


if __name__ == '__main__':
    import imageio
    import tqdm
    img = cv2.imread("assets/cano_secc.png")
    img = img / 127.5 - 1
    img = torch.FloatTensor(img).permute(2, 0, 1)
    fps = 25
    writer = imageio.get_writer('demo_blink.mp4', fps=fps)

    for i in tqdm.trange(33):
        blink_percent = 0.03 * i
        with Timer("Blink", True):
            out_img = blink_eye_for_secc(img, blink_percent)
        out_img = ((out_img.permute(1,2,0)+1)*127.5).int().numpy()
        writer.append_data(out_img)
    writer.close()