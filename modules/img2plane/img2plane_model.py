import torch
import torch.nn as nn
import torch.nn.functional as F

from .deeplabv3 import DeepLabV3
from .simple_encoders.high_resolution_encoder import HighResoEncoder
from .segformer import LowResolutionViT, TriplanePredictorViT
import copy
from utils.commons.hparams import hparams


class Img2PlaneModel(nn.Module):
    def __init__(self, out_channels=96, hp=None):
        super().__init__()
        global hparams
        self.hparams = hp if hp is not None else copy.deepcopy(hparams)
        hparams = self.hparams
        
        self.input_mode = hparams.get("img2plane_input_mode", "rgb")
        if self.input_mode == 'rgb':
            in_channels = 3 
        elif self.input_mode == 'rgb_alpha':
            in_channels = 4 
        elif self.input_mode == 'rgb_camera':
            self.camera_to_channel = nn.Linear(25, 3)
            in_channels = 3 + 3
        elif self.input_mode == 'rgb_alpha_camera':
            self.camera_to_channel = nn.Linear(25, 3)
            in_channels = 4 + 3

        in_channels += 2 # add grid_x and grid_y, act as positional encoding
        self.low_reso_encoder = DeepLabV3(in_channels=in_channels)
        self.high_reso_encoder = HighResoEncoder(in_dim=in_channels)
        self.low_reso_vit = LowResolutionViT()
        self.triplane_predictor_vit = TriplanePredictorViT(out_channels=out_channels, img2plane_backbone_scale=hparams['img2plane_backbone_scale'])

    def forward(self, x, cond=None, **synthesis_kwargs):
        """
        x: original image, [B, 3, H=512, W=512] 
        return: predicted triplane, [B, 32*3, H=256, W=256]
        optional:
            ref_alphas: 0/1 mask, if img2plane, all ones; if secc2plane, only ones for head, [B, 1, H, W]
            ref_camera: camera pose of the input img, [B, 25]
        """
        bs, _, H, W = x.shape

        if self.input_mode in ['rgb_alpha', 'rgb_alpha_camera']:
            if cond is None or cond.get("ref_alphas") is None:
                ref_alphas = (x.mean(dim=1, keepdim=True) >= -0.999).float() # set non-black to ones
            else:
                ref_alphas = cond["ref_alphas"]
            x = torch.cat([x, ref_alphas], dim=1)
        if self.input_mode in ['rgb_camera', 'rgb_alpha_camera']:
            ref_cameras = cond["ref_cameras"]
            camera_feat = self.camera_to_channel(ref_cameras).reshape(bs, 3, 1, 1).repeat([1, 1, H, W])
            x = torch.cat([x, camera_feat], dim=1)

        # concat with pixel position
        grid_x, grid_y = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device)) # [H, W]
        grid_x = grid_x / H
        grid_y = grid_y / H
        expanded_x = grid_x[None, None, :, :].repeat(bs, 1, 1, 1) # [B, 1, H, W]
        expanded_y = grid_y[None, None, :, :].repeat(bs, 1, 1, 1) # [B, 1, H, W]
        x = torch.cat([x, expanded_x, expanded_y], dim=1) # [B, 3+1+1, H, W]

        feat_low = self.low_reso_encoder(x)
        feat_low_after_vit = self.low_reso_vit(feat_low)
        feat_high = self.high_reso_encoder(x)
        # self.triplane_predictor_vit OUTCHANNEL *4, VIEW 4,3,-1, FLIP 注意dim的idx
        planes = self.triplane_predictor_vit(feat_low_after_vit, feat_high) # [B, C, H, W]

        planes = planes.view(len(planes), 3, -1, planes.shape[-2], planes.shape[-1])

        # borrowed from img2plane and hide-nerf
        planes_xy = planes[:,0]
        planes_xy = torch.flip(planes_xy, [2])
        planes_xz = planes[:,1]
        planes_xz = torch.flip(planes_xz, [2])
        planes_zy = planes[:,2]
        planes_zy = torch.flip(planes_zy, [2, 3])
        planes = torch.stack([planes_xy, planes_xz, planes_zy], dim=1) # [N, 3, C, H, W]
        return planes