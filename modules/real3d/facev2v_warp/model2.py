import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
import copy 

from modules.real3d.facev2v_warp.network2 import AppearanceFeatureExtractor, CanonicalKeypointDetector, PoseExpressionEstimator, MotionFieldEstimator, Generator
from modules.real3d.facev2v_warp.func_utils import transform_kp, make_coordinate_grid_2d, apply_imagenet_normalization
from modules.real3d.facev2v_warp.losses import PerceptualLoss, GANLoss, FeatureMatchingLoss, EquivarianceLoss, KeypointPriorLoss, HeadPoseLoss, DeformationPriorLoss
from utils.commons.image_utils import erode, dilate
from utils.commons.hparams import hparams


class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)
        self.idx_tensor = torch.FloatTensor(list(range(num_bins))).unsqueeze(0).cuda()
        self.n_bins = num_bins
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        real_yaw = self.fc_yaw(x)
        real_pitch = self.fc_pitch(x)
        real_roll = self.fc_roll(x)
        real_yaw = torch.softmax(real_yaw, dim=1)
        real_pitch = torch.softmax(real_pitch, dim=1)
        real_roll = torch.softmax(real_roll, dim=1)
        real_yaw = (real_yaw * self.idx_tensor).sum(dim=1)
        real_pitch = (real_pitch * self.idx_tensor).sum(dim=1)
        real_roll = (real_roll * self.idx_tensor).sum(dim=1)
        real_yaw = (real_yaw - self.n_bins // 2) * 3 * np.pi / 180
        real_pitch = (real_pitch - self.n_bins // 2) * 3 * np.pi / 180
        real_roll = (real_roll - self.n_bins // 2) * 3 * np.pi / 180

        return real_yaw, real_pitch, real_roll


class Transform:
    """
    Random tps transformation for equivariance constraints.
    reference: FOMM
    """

    def __init__(self, bs, sigma_affine=0.05, sigma_tps=0.005, points_tps=5):
        noise = torch.normal(mean=0, std=sigma_affine * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        self.control_points = make_coordinate_grid_2d((points_tps, points_tps))
        self.control_points = self.control_points.unsqueeze(0)
        self.control_params = torch.normal(mean=0, std=sigma_tps * torch.ones([bs, 1, points_tps ** 2]))

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:]).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, align_corners=True, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        control_points = self.control_points.type(coordinates.type())
        control_params = self.control_params.type(coordinates.type())
        distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
        distances = torch.abs(distances).sum(-1)

        result = distances ** 2
        result = result * torch.log(distances + 1e-6)
        result = result * control_params
        result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
        transformed = transformed + result

        return transformed


class WarpBasedTorsoModel(nn.Module):
    def __init__(self, model_scale='small'):
        super().__init__()
        self.appearance_extractor = AppearanceFeatureExtractor(model_scale)
        self.canonical_kp_detector = CanonicalKeypointDetector(model_scale)
        self.pose_exp_estimator = PoseExpressionEstimator(model_scale)
        self.motion_field_estimator = MotionFieldEstimator(model_scale)
        self.deform_based_generator = Generator()

        self.pretrained_hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=66).cuda()
        pretrained_path = "/home/tiger/nfs/myenv/cache/useful_ckpts/hopenet_robust_alpha1.pkl" # https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR
        self.pretrained_hopenet.load_state_dict(torch.load(pretrained_path, map_location=torch.device("cpu")))
        self.pretrained_hopenet.requires_grad_(False)

        self.pose_loss_fn = HeadPoseLoss() # 20
        self.equivariance_loss_fn = EquivarianceLoss() # 20
        self.keypoint_prior_loss_fn = KeypointPriorLoss()# 10
        self.deform_prior_loss_fn = DeformationPriorLoss() # 5

    def forward(self, torso_src_img, src_img, drv_img, cal_loss=False):
        # predict cano keypoint
        cano_keypoint = self.canonical_kp_detector(src_img)
        # predict src_pose and drv_pose
        transform_fn = Transform(drv_img.shape[0])
        transformed_drv_img = transform_fn.transform_frame(drv_img)
        cat_imgs = torch.cat([src_img, drv_img, transformed_drv_img], dim=0)
        yaw, pitch, roll, t, delta = self.pose_exp_estimator(cat_imgs)
        [yaw_s, yaw_d, yaw_tran], [pitch_s, pitch_d, pitch_tran], [roll_s, roll_d, roll_tran] = (
            torch.chunk(yaw, 3, dim=0),
            torch.chunk(pitch, 3, dim=0),
            torch.chunk(roll, 3, dim=0),
        )
        [t_s, t_d, t_tran], [delta_s, delta_d, delta_tran] = (
            torch.chunk(t, 3, dim=0),
            torch.chunk(delta, 3, dim=0),
        )
        kp_s, Rs = transform_kp(cano_keypoint, yaw_s, pitch_s, roll_s, t_s, delta_s)
        kp_d, Rd = transform_kp(cano_keypoint, yaw_d, pitch_d, roll_d, t_d, delta_d)
        # deform the torso img
        torso_appearance_feats = self.appearance_extractor(torso_src_img)
        deformation, occlusion = self.motion_field_estimator(torso_appearance_feats, kp_s, kp_d, Rs, Rd)
        deformed_torso_img = self.deform_based_generator(torso_appearance_feats, deformation, occlusion)
        
        ret = {'kp_src': kp_s, 'kp_drv': kp_d}
        if cal_loss:
            losses = {}
            with torch.no_grad():
                self.pretrained_hopenet.eval()
                real_yaw, real_pitch, real_roll = self.pretrained_hopenet(F.interpolate(apply_imagenet_normalization(cat_imgs), size=(224, 224)))
            pose_loss = self.pose_loss_fn(yaw, pitch, roll, real_yaw, real_pitch, real_roll)
            losses['facev2v/pose_pred_loss'] = pose_loss

            kp_tran, _ = transform_kp(cano_keypoint, yaw_tran, pitch_tran, roll_tran, t_tran, delta_tran)
            reverse_kp = transform_fn.warp_coordinates(kp_tran[:, :, :2])
            equivariance_loss = self.equivariance_loss_fn(kp_d, reverse_kp)
            losses['facev2v/equivariance_loss'] = equivariance_loss

            keypoint_prior_loss = self.keypoint_prior_loss_fn(kp_d)
            losses['facev2v/keypoint_prior_loss'] = keypoint_prior_loss

            deform_prior_loss = self.deform_prior_loss_fn(delta_d)
            losses['facev2v/deform_prior_loss'] = deform_prior_loss
            ret['losses'] = losses

        return deformed_torso_img, ret


class WarpBasedTorsoModelMediaPipe(nn.Module):
    def __init__(self, model_scale='small'):
        super().__init__()
        self.hparams = copy.deepcopy(hparams)
        if hparams.get("torso_inp_mode", "rgb") == 'rgb_alpha':
            torso_in_dim = 5
        else:
            torso_in_dim = 3
        self.appearance_extractor = AppearanceFeatureExtractor(in_dim=torso_in_dim, model_scale=model_scale)
        self.motion_field_estimator = MotionFieldEstimator(model_scale, input_channels=32+2, num_keypoints=self.hparams['torso_kp_num']) # 32 channel appearance channel, and 3 channel for segmap
        # self.motion_field_estimator = MotionFieldEstimator(model_scale, input_channels=32+2, num_keypoints=9) # 32 channel appearance channel, and 3 channel for segmap
        self.deform_based_generator = Generator()

        self.occlusion_2_predictor = nn.Sequential(*[
            nn.Conv2d(64+1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        ])

    #  V2, 先warp， 再mean
    def forward(self, torso_src_img, segmap, kp_s, kp_d, tgt_head_img, tgt_head_weights, cal_loss=False, target_torso_mask=None):
        """
        kp_s, kp_d, [b, 68, 3], within the range of [-1,1]
        """
        if hparams.get("torso_inp_mode", "rgb") == 'rgb_alpha':
            torso_segmap = torch.nn.functional.interpolate(segmap[:,[2,4]].float(), size=(torso_src_img.shape[-2],torso_src_img.shape[-1]), mode='bilinear', align_corners=False, antialias=False) # see tasks/eg3ds/loss_utils/segment_loss/mp_segmenter.py for the segmap convention
            torso_src_img = torch.cat([torso_src_img, torso_segmap], dim=1)

        torso_appearance_feats = self.appearance_extractor(torso_src_img) # [B, C, D, H, W]
        torso_segmap = torch.nn.functional.interpolate(segmap[:,[2,4]].float(), size=(64,64), mode='bilinear', align_corners=False, antialias=False) # see tasks/eg3ds/loss_utils/segment_loss/mp_segmenter.py for the segmap convention
        torso_mask = torso_segmap.sum(dim=1).unsqueeze(1) # [b, 1, ,h, w]
        torso_mask = dilate(torso_mask, ksize=self.hparams.get("torso_mask_dilate_ksize", 7))
        if self.hparams.get("mul_torso_mask", True):
            torso_appearance_feats = torso_appearance_feats * torso_mask.unsqueeze(1)
        motion_inp_appearance_feats = torch.cat([torso_appearance_feats, torso_segmap.unsqueeze(2).repeat([1,1,torso_appearance_feats.shape[2],1,1])], dim=1)

        if self.hparams['torso_kp_num'] == 4:
            kp_s = kp_s[:,[0,8,16,27],:]
            kp_d = kp_d[:,[0,8,16,27],:]
        elif self.hparams['torso_kp_num'] == 9:
            kp_s = kp_s[:,[0, 3, 6, 8, 10, 13, 16, 27, 33],:]
            kp_d = kp_d[:,[0, 3, 6, 8, 10, 13, 16, 27, 33],:]
        else:
            raise NotImplementedError()

        # deform the torso img
        Rs = torch.eye(3, 3).unsqueeze(0).repeat([kp_s.shape[0], 1, 1]).to(kp_s.device)
        Rd = torch.eye(3, 3).unsqueeze(0).repeat([kp_d.shape[0], 1, 1]).to(kp_d.device)
        deformation, occlusion, occlusion_2 = self.motion_field_estimator(motion_inp_appearance_feats, kp_s, kp_d, Rs, Rd, tgt_head_img, tgt_head_weights)
        motion_estimator_grad_scale_factor = 0.1
        # motion_estimator_grad_scale_factor = 1.0
        deformation = deformation * motion_estimator_grad_scale_factor + deformation.detach() * (1-motion_estimator_grad_scale_factor)
        # occlusion, a 0~1 mask that predict the segment map of warped torso, used in oclcusion-aware decoder
        occlusion = occlusion * motion_estimator_grad_scale_factor + occlusion.detach() * (1-motion_estimator_grad_scale_factor)
        # occlusion_2, a 0~1 mask that predict the segment map of warped torso, but is used in alpha-blending
        occlusion_2 = occlusion_2 * motion_estimator_grad_scale_factor + occlusion_2.detach() * (1-motion_estimator_grad_scale_factor)
        ret = {'kp_src': kp_s, 'kp_drv': kp_d, 'occlusion': occlusion, 'occlusion_2': occlusion_2}

        deformed_torso_img, deformed_torso_hid = self.deform_based_generator(torso_appearance_feats, deformation, occlusion, return_hid=True)
        ret['deformed_torso_hid'] = deformed_torso_hid
        occlusion_2 = self.occlusion_2_predictor(torch.cat([deformed_torso_hid, F.interpolate(occlusion_2, size=(256,256), mode='bilinear')], dim=1))
        ret['occlusion_2'] = occlusion_2
        alphas = occlusion_2.clamp(1e-5, 1 - 1e-5) 

        if target_torso_mask is None:
            ret['losses'] = {
                'facev2v/occlusion_reg_l1': occlusion.mean(),
                'facev2v/occlusion_2_reg_l1': occlusion_2.mean(),
                'facev2v/occlusion_2_weights_entropy': torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)), # you can visualize this fn at https://www.desmos.com/calculator/rwbs7bruvj?lang=zh-TW
            }
        else:
            non_target_torso_mask_1 = torch.nn.functional.interpolate((~target_torso_mask).unsqueeze(1).float(), size=occlusion.shape[-2:])
            non_target_torso_mask_2 = torch.nn.functional.interpolate((~target_torso_mask).unsqueeze(1).float(), size=occlusion_2.shape[-2:])
            ret['losses'] = {
                'facev2v/occlusion_reg_l1': self.masked_l1_reg_loss(occlusion, non_target_torso_mask_1.bool(), masked_weight=1, unmasked_weight=self.hparams['torso_occlusion_reg_unmask_factor']),
                'facev2v/occlusion_2_reg_l1': self.masked_l1_reg_loss(occlusion_2, non_target_torso_mask_2.bool(), masked_weight=1, unmasked_weight=self.hparams['torso_occlusion_reg_unmask_factor']),
                'facev2v/occlusion_2_weights_entropy': torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)), # you can visualize this fn at https://www.desmos.com/calculator/rwbs7bruvj?lang=zh-TW
            }
        # if self.hparams.get("fuse_with_deform_source"):
        #     B, _, H, W = deformed_torso_img.shape
        #     deformation_256 = F.interpolate(deformation.mean(dim=1).permute(0,3,1,2), size=256, mode='bilinear',antialias=True).permute(0,2,3,1)[...,:2]
        #     deformed_source_torso_img = F.grid_sample(torso_src_img, deformation_256, align_corners=True).view(B, -1, H, W)
        #     occlusion_256 = F.interpolate(occlusion, size=256, antialias=True, mode='bilinear').reshape([B,1,H,W])
        #     # deformed_torso_img = deformed_torso_img * (1 - occlusion_256[:,0]) + deformed_source_torso_img[:,0] * occlusion_256[:,0]
        #     deformed_torso_img = deformed_torso_img * (1 - occlusion_256) + deformed_source_torso_img * occlusion_256
        return deformed_torso_img, ret

    def masked_l1_reg_loss(self, img_pred, mask, masked_weight=0.01, unmasked_weight=0.001, mode='l1'):
        # 对raw图像，因为deform的原因背景没法全黑，导致这部分mse过高，我们将其mask掉，只计算人脸部分
        masked_weight = 1.0
        weight_mask = mask.float() * masked_weight + (~mask).float() * unmasked_weight
        if mode == 'l1':
            error = (img_pred).abs().sum(dim=1) * weight_mask
        else:
            error = (img_pred).pow(2).sum(dim=1) * weight_mask
        loss = error.mean()
        return loss

    @torch.no_grad()
    def infer_forward_stage1(self, torso_src_img, segmap, kp_s, kp_d, tgt_head_img, cal_loss=False):
        """
        kp_s, kp_d, [b, 68, 3], within the range of [-1,1]
        """
        kp_s = kp_s[:,[0,8,16,27],:]
        kp_d = kp_d[:,[0,8,16,27],:]

        torso_segmap = torch.nn.functional.interpolate(segmap[:,[2,4]].float(), size=(64,64), mode='bilinear', align_corners=False, antialias=False) # see tasks/eg3ds/loss_utils/segment_loss/mp_segmenter.py for the segmap convention
        torso_appearance_feats = self.appearance_extractor(torso_src_img)
        torso_mask = torso_segmap.sum(dim=1).unsqueeze(1) # [b, 1, ,h, w]
        torso_mask = dilate(torso_mask, ksize=self.hparams.get("torso_mask_dilate_ksize", 7))
        if self.hparams.get("mul_torso_mask", True):
            torso_appearance_feats = torso_appearance_feats * torso_mask.unsqueeze(1)
        motion_inp_appearance_feats = torch.cat([torso_appearance_feats, torso_segmap.unsqueeze(2).repeat([1,1,torso_appearance_feats.shape[2],1,1])], dim=1)
        # deform the torso img
        Rs = torch.eye(3, 3).unsqueeze(0).repeat([kp_s.shape[0], 1, 1]).to(kp_s.device)
        Rd = torch.eye(3, 3).unsqueeze(0).repeat([kp_d.shape[0], 1, 1]).to(kp_d.device)
        deformation, occlusion, occlusion_2 = self.motion_field_estimator(motion_inp_appearance_feats, kp_s, kp_d, Rs, Rd)
        motion_estimator_grad_scale_factor = 0.1
        deformation = deformation * motion_estimator_grad_scale_factor + deformation.detach() * (1-motion_estimator_grad_scale_factor)
        occlusion = occlusion * motion_estimator_grad_scale_factor + occlusion.detach() * (1-motion_estimator_grad_scale_factor)
        occlusion_2 = occlusion_2 * motion_estimator_grad_scale_factor + occlusion_2.detach() * (1-motion_estimator_grad_scale_factor)
        ret = {'kp_src': kp_s, 'kp_drv': kp_d, 'occlusion': occlusion, 'occlusion_2': occlusion_2}
        ret['torso_appearance_feats'] = torso_appearance_feats
        ret['deformation'] = deformation
        ret['occlusion'] = occlusion
        return ret
    
    @torch.no_grad()
    def infer_forward_stage2(self, ret):
        torso_appearance_feats = ret['torso_appearance_feats']
        deformation = ret['deformation']
        occlusion = ret['occlusion']
        deformed_torso_img, deformed_torso_hid = self.deform_based_generator(torso_appearance_feats, deformation, occlusion, return_hid=True)
        ret['deformed_torso_hid'] = deformed_torso_hid
        return deformed_torso_img
    
if __name__ == '__main__':
    from utils.nn.model_utils import num_params
    import tqdm
    model = WarpBasedTorsoModel('small')
    model.cuda()
    num_params(model)
    for n, m in model.named_children():
        num_params(m, model_name=n)
    torso_ref_img = torch.randn([2, 3, 256, 256]).cuda()
    ref_img = torch.randn([2, 3, 256, 256]).cuda()
    mv_img = torch.randn([2, 3, 256, 256]).cuda()
    out = model(torso_ref_img, ref_img, mv_img)
    for i in tqdm.trange(100):
        out_img, losses = model(torso_ref_img, ref_img, mv_img, cal_loss=True)
    print(" ")