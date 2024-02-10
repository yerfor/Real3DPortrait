import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from modules.real3d.facev2v_warp.layers import ConvBlock2D, DownBlock2D, DownBlock3D, UpBlock2D, UpBlock3D, ResBlock2D, ResBlock3D, ResBottleneck
from modules.real3d.facev2v_warp.func_utils import (
    out2heatmap,
    heatmap2kp,
    kp2gaussian_2d,
    create_heatmap_representations,
    create_sparse_motions,
    create_deformed_source_image,
)

class AppearanceFeatureExtractor(nn.Module):
    # 3D appearance features extractor
    # [N,3,256,256]
    # [N,64,256,256]
    # [N,128,128,128]
    # [N,256,64,64]
    # [N,512,64,64]
    # [N,32,16,64,64]
    def __init__(self, in_dim=3, model_scale='standard', lora_args=None):
        super().__init__()
        use_weight_norm = False
        down_seq = [64, 128, 256]
        n_res = 6
        C = 32
        D = 16
        self.in_conv = ConvBlock2D("CNA", in_dim, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], C * D, 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock3D(C, use_weight_norm) for _ in range(n_res)])

        self.C, self.D = C, D

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.mid_conv(x)
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W)
        x = self.res(x)
        return x


class CanonicalKeypointDetector(nn.Module):
    # Canonical keypoints detector
    # [N,3,256,256]
    # [N,64,128,128]
    # [N,128,64,64]
    # [N,256,32,32]
    # [N,512,16,16]
    # [N,1024,8,8]
    # [N,16384,8,8]
    # [N,1024,16,8,8]
    # [N,512,16,16,16]
    # [N,256,16,32,32]
    # [N,128,16,64,64]
    # [N,64,16,128,128]
    # [N,32,16,256,256]
    # [N,20,16,256,256] (heatmap)
    # [N,20,3] (key points)
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm=False

        if model_scale == 'standard' or model_scale == 'large':
            down_seq = [3, 64, 128, 256, 512, 1024]
            up_seq = [1024, 512, 256, 128, 64, 32]
            D = 16 # depth_channel 
            K = 15
            scale_factor=0.25
        elif model_scale == 'small':
            down_seq = [3, 32, 64, 128, 256, 512]
            up_seq = [512, 256, 128, 64, 32, 16]
            D = 6 # depth_channel 
            K = 15
            scale_factor=0.25
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.mid_conv = nn.Conv2d(down_seq[-1], up_seq[0] * D, 1, 1, 0)
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])
        self.out_conv = nn.Conv3d(up_seq[-1], K, 3, 1, 1)
        self.C, self.D = up_seq[0], D
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, mode="bilinear", scale_factor=self.scale_factor, align_corners=False, recompute_scale_factor=True)
        # [1, 3, 256, 256] ==> [1, 3, 64, 64]
        x = self.down(x) # ==> [1, 1024, 2, 2]
        x = self.mid_conv(x) # ==> [1, 16384, 2, 2]
        N, _, H, W = x.shape
        x = x.view(N, self.C, self.D, H, W) # ==> [1, 1024, 16, 2, 2]
        x = self.up(x) # ==> [1, 32, 16, 64, 64]
        x = self.out_conv(x) # ==> [1, 15, 16, 64, 64]
        heatmap = out2heatmap(x)
        kp = heatmap2kp(heatmap)
        return kp


class PoseExpressionEstimator(nn.Module):
    # Head pose estimator && expression deformation estimator
    # [N,3,256,256]
    # [N,64,64,64]
    # [N,256,64,64]
    # [N,512,32,32]
    # [N,1024,16,16]
    # [N,2048,8,8]
    # [N,2048]
    # [N,66] [N,66] [N,66] [N,3] [N,60]
    # [N,] [N,] [N,] [N,3] [N,20,3]
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm=False
        n_bins=66
        K=15
        if model_scale == 'standard' or model_scale == 'large':
            n_filters=[64, 256, 512, 1024, 2048]
            n_blocks=[3, 3, 5, 2]
        elif model_scale == 'small':
            n_filters=[32, 128, 256, 512, 512]
            n_blocks=[2, 2, 4, 2]

        self.pre_layers = nn.Sequential(ConvBlock2D("CNA", 3, n_filters[0], 7, 2, 3, use_weight_norm), nn.MaxPool2d(3, 2, 1))
        res_layers = []
        for i in range(len(n_filters) - 1):
            res_layers.extend(self._make_layer(i, n_filters[i], n_filters[i + 1], n_blocks[i], use_weight_norm))
        self.res_layers = nn.Sequential(*res_layers)
        self.fc_yaw = nn.Linear(n_filters[-1], n_bins)
        self.fc_pitch = nn.Linear(n_filters[-1], n_bins)
        self.fc_roll = nn.Linear(n_filters[-1], n_bins)
        self.fc_t = nn.Linear(n_filters[-1], 3)
        self.fc_delta = nn.Linear(n_filters[-1], 3 * K)
        self.n_bins = n_bins
        self.idx_tensor = torch.FloatTensor(list(range(self.n_bins))).unsqueeze(0).cuda()

    def _make_layer(self, i, in_channels, out_channels, n_block, use_weight_norm):
        stride = 1 if i == 0 else 2
        return [ResBottleneck(in_channels, out_channels, stride, use_weight_norm)] + [
            ResBottleneck(out_channels, out_channels, 1, use_weight_norm) for _ in range(n_block)
        ]

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = torch.mean(x, (2, 3))
        yaw, pitch, roll, t, delta = self.fc_yaw(x), self.fc_pitch(x), self.fc_roll(x), self.fc_t(x), self.fc_delta(x)
        yaw = torch.softmax(yaw, dim=1)
        pitch = torch.softmax(pitch, dim=1)
        roll = torch.softmax(roll, dim=1)
        yaw = (yaw * self.idx_tensor).sum(dim=1)
        pitch = (pitch * self.idx_tensor).sum(dim=1)
        roll = (roll * self.idx_tensor).sum(dim=1)
        yaw = (yaw - self.n_bins // 2) * 3 * np.pi / 180
        pitch = (pitch - self.n_bins // 2) * 3 * np.pi / 180
        roll = (roll - self.n_bins // 2) * 3 * np.pi / 180
        delta = delta.view(x.shape[0], -1, 3)
        return yaw, pitch, roll, t, delta


class MotionFieldEstimator(nn.Module):
    # Motion field estimator
    # (4+1)x(20+1)=105
    # [N,105,16,64,64]
    # ...
    # [N,32,16,64,64]
    # [N,137,16,64,64]
    # 1.
    # [N,21,16,64,64] (mask)
    # 2.
    # [N,2192,64,64]
    # [N,1,64,64] (occlusion)
    def __init__(self, model_scale='standard', input_channels=32, num_keypoints=15, predict_multiref_occ=True):
        super().__init__()
        use_weight_norm=False
        if model_scale == 'standard' or model_scale == 'large':
            down_seq = [(num_keypoints+1)*5, 64, 128, 256, 512, 1024]
            up_seq = [1024, 512, 256, 128, 64, 32]
        elif model_scale == 'small':
            down_seq = [(num_keypoints+1)*5, 32, 64, 128, 256, 512]
            up_seq = [512, 256, 128, 64, 32, 16]
        K = num_keypoints
        D = 16
        C1 = input_channels # appearance feats channel
        C2 = 4
        self.compress = nn.Conv3d(C1, C2, 1, 1, 0)
        self.down = nn.Sequential(*[DownBlock3D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.up = nn.Sequential(*[UpBlock3D(up_seq[i], up_seq[i + 1], use_weight_norm) for i in range(len(up_seq) - 1)])

        tgt_head_in_dim = 3 + 1
        tgt_head_hid_dim = 32
        tgt_head_layers =  [ConvBlock2D("CNA", tgt_head_in_dim, tgt_head_hid_dim, 7, 1, 3, use_weight_norm)] + [ResBlock2D(tgt_head_hid_dim, use_weight_norm) for _ in range(3)]
        self.tgt_head_encoder = nn.Sequential(*tgt_head_layers)
        self.tgt_head_fuser = nn.Conv3d(tgt_head_hid_dim + down_seq[0] + up_seq[-1], tgt_head_hid_dim, 7, 1, 3)
        
        self.mask_conv = nn.Conv3d(tgt_head_hid_dim, K + 1, 7, 1, 3)
        self.predict_multiref_occ = predict_multiref_occ
        self.occlusion_conv = nn.Conv2d(tgt_head_hid_dim * D, 1, 7, 1, 3)
        self.occlusion_conv2 = nn.Conv2d(tgt_head_hid_dim * D, 1, 7, 1, 3)

        self.C, self.D = down_seq[0] + up_seq[-1], D
        
    def forward(self, fs, kp_s, kp_d, Rs, Rd, tgt_head_img, tgt_head_weights):
        # the original fs is compressed to 4 channels using a 1x1x1 conv
        fs_compressed = self.compress(fs)
        N, _, D, H, W = fs.shape
        # [N,21,1,16,64,64]
        heatmap_representation = create_heatmap_representations(fs_compressed, kp_s, kp_d)
        # [N,21,16,64,64,3]
        sparse_motion = create_sparse_motions(fs_compressed, kp_s, kp_d, Rs, Rd)
        # [N,21,4,16,64,64]
        deformed_source = create_deformed_source_image(fs_compressed, sparse_motion)
        input = torch.cat([heatmap_representation, deformed_source], dim=2).view(N, -1, D, H, W)
        output = self.down(input)
        output = self.up(output)
        x = torch.cat([input, output], dim=1)

        tgt_head_inp = torch.cat([tgt_head_img, tgt_head_weights], dim=1)
        tgt_head_inp = torch.nn.functional.interpolate(tgt_head_inp, size=(128,128), mode='bilinear')
        tgt_head_feats = self.tgt_head_encoder(tgt_head_inp) # [B, C=3+1, H=256, W=256]
        tgt_head_feats = torch.nn.functional.interpolate(tgt_head_feats, size=(64,64), mode='bilinear')

        fused_x = torch.cat([x, tgt_head_feats.unsqueeze(2).repeat([1,1,x.shape[2],1,1])], dim=1)
        x = self.tgt_head_fuser(fused_x)

        mask = self.mask_conv(x)
        # [N,21,16,64,64,1]
        mask = F.softmax(mask, dim=1).unsqueeze(-1)
        # [N,16,64,64,3]
        deformation = (sparse_motion * mask).sum(dim=1)
        if self.predict_multiref_occ:
            occlusion, occlusion_2 = self.create_occlusion(x.view(N, -1, H, W))
            return deformation, occlusion, occlusion_2
        else:
            return deformation, x.view(N, -1, H, W)
        
    # x: torch.Tensor, N, M, H, W
    def create_occlusion(self, x, deformed_source=None):
        occlusion = self.occlusion_conv(x)
        occlusion_2 = self.occlusion_conv2(x)
        occlusion = torch.sigmoid(occlusion)
        occlusion_2 = torch.sigmoid(occlusion_2)
        return occlusion, occlusion_2
    


class Generator(nn.Module):
    # Generator
    # [N,32,16,64,64]
    # [N,512,64,64]
    # [N,256,64,64]
    # [N,128,128,128]
    # [N,64,256,256]
    # [N,3,256,256]
    def __init__(self, input_channels=32, model_scale='standard', more_res=False):
        super().__init__()
        use_weight_norm=True
        C=input_channels
        
        if model_scale == 'large':
            n_res = 12
            up_seq = [256, 128, 64]
            D = 16
            use_up_res = True
        elif model_scale in ['standard', 'small']:
            n_res = 6
            up_seq = [256, 128, 64]
            D = 16 
            use_up_res = False
        self.in_conv = ConvBlock2D("CNA", C * D, up_seq[0], 3, 1, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.mid_conv = nn.Conv2d(up_seq[0], up_seq[0], 1, 1, 0)
        self.res = nn.Sequential(*[ResBlock2D(up_seq[0], use_weight_norm) for _ in range(n_res)])
        ups = []
        for i in range(len(up_seq) - 1):
            ups.append(UpBlock2D(up_seq[i], up_seq[i + 1], use_weight_norm))
            if use_up_res:
                ups.append(ResBlock2D(up_seq[i + 1], up_seq[i + 1]))
        self.up = nn.Sequential(*ups)
        self.out_conv = nn.Conv2d(up_seq[-1], 3, 7, 1, 3)
               
    def forward(self, fs, deformation, occlusion, return_hid=False):
        deformed_fs = self.get_deformed_feature(fs, deformation)
        return self.forward_with_deformed_feature(deformed_fs, occlusion, return_hid=return_hid)
    
    def forward_with_deformed_feature(self, deformed_fs, occlusion, return_hid=False):
        fs = deformed_fs
        fs = self.in_conv(fs)
        fs = self.mid_conv(fs)
        fs = self.res(fs)
        fs = self.up(fs)
        rgb = self.out_conv(fs)
        if return_hid:
            return rgb, fs
        return rgb
    
    @staticmethod
    def get_deformed_feature(fs, deformation):
        N, _, D, H, W = fs.shape
        fs = F.grid_sample(fs, deformation, align_corners=True, padding_mode='border').view(N, -1, H, W)
        return fs


class Discriminator(nn.Module):
    # Patch Discriminator

    def __init__(self, use_weight_norm=True, down_seq=[64, 128, 256, 512], K=15):
        super().__init__()
        layers = []
        layers.append(ConvBlock2D("CNA", 3 + K, down_seq[0], 3, 2, 1, use_weight_norm, "instance", "leakyrelu"))
        layers.extend(
            [
                ConvBlock2D("CNA", down_seq[i], down_seq[i + 1], 3, 2 if i < len(down_seq) - 2 else 1, 1, use_weight_norm, "instance", "leakyrelu")
                for i in range(len(down_seq) - 1)
            ]
        )
        layers.append(ConvBlock2D("CN", down_seq[-1], 1, 3, 1, 1, use_weight_norm, activation_type="none"))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, kp):
        heatmap = kp2gaussian_2d(kp.detach()[:, :, :2], x.shape[2:])
        x = torch.cat([x, heatmap], dim=1)
        res = [x]
        for layer in self.layers:
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1:-1]
        return output, features
