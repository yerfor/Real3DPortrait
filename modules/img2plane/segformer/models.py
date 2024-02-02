import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .base import OverlapPatchEmbed, Block
from utils.commons.hparams import hparams

class LowResolutionViT(nn.Module):
    """
    This Vit process the output of low resolution image features produced by DeepLabv3
    """
    def __init__(self, img_size=64, in_chans=256):
        super().__init__()

        # patch_embed
        self.patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=3, stride=2, in_chans=in_chans, embed_dim=1024)
        
        if hparams.get('img2plane_backbone_scale', 'standard') == 'small':
            self.num_blocks = 2
        if hparams.get('img2plane_backbone_scale', 'standard') == 'standard':
            self.num_blocks = 5
        elif hparams['img2plane_backbone_scale'] == 'large':
            self.num_blocks = 10
        for i in range(1, self.num_blocks+1):
            setattr(self, f'block{i}', Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1))
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.upsampling_bilinear1 = nn.UpsamplingBilinear2d(scale_factor=2.)
        self.conv_after_upsample1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.activation_conv1 = nn.ReLU()
        self.upsampling_bilinear2 = nn.UpsamplingBilinear2d(scale_factor=2.)
        self.conv_after_upsample2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.activation_conv2 = nn.ReLU()
        self.final_conv = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def freeze_patch_emb(self):
        self.patch_embed.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}  # has pos_embed may be better

    def forward(self, x):
        """
        x: [B, 256, 64, 64]
        return [B, C=96, H=256, W=256]
        """
        h, H, W = self.patch_embed(x)

        for i in range(1, self.num_blocks+1):
            block_i = getattr(self, f'block{i}')
            h = block_i(h, H, H) # [B=2, 1024, H*W=1024]

        h = h.permute(0, 2, 1) # [B, C, N=H*W]
        h = h.view(h.shape[0], h.shape[1], H, W) # [B=2, C=1024, H=32, W=32]

        h = self.pixel_shuffle(h) # [B=2, C=256, H=64, W=64]
        h = self.upsampling_bilinear1(h) # [B=2, C=256, H=128, W=128]
        h = self.conv_after_upsample1(h)
        h = self.activation_conv1(h)
        h = self.upsampling_bilinear2(h) # [B=2, C, H=256, W=256]
        h = self.conv_after_upsample2(h)
        h = self.activation_conv2(h)
        
        out = self.final_conv(h)
        return out


class TriplanePredictorViT(nn.Module):
    """
    This Vit process the concatenated features of LowResolutionViT and the CNN-based HighResoEncoder
    It predicts the final Tri-plane!
    """
    def __init__(self, img_size=256, out_channels=96, img2plane_backbone_scale='standard'):
        super().__init__()
        # the input is concated features, 96 from low_reso_vit and 96 from high_resolution encoder
        self.first_conv = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.second_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=3, stride=2, in_chans=128, embed_dim=1024)

        if img2plane_backbone_scale == 'small':
            self.num_blocks = 1
        if img2plane_backbone_scale == 'standard':
            self.num_blocks = 1
        elif img2plane_backbone_scale == 'large':
            self.num_blocks = 3
        for i in range(1, self.num_blocks+1):
            # sr_ratio = 2 if i == 1 else 1
            sr_ratio = 2
            setattr(self, f'block{i}', Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=sr_ratio))
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        # skip concat with low resolution, 256 from pixel_shuffle + 96 from low_reso_vit
        self.first_conv_after_cat = nn.Conv2d(in_channels=352, out_channels=256, kernel_size=3, stride=1, padding=1) 
        self.second_conv_after_cat = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1) 
        self.third_conv_after_cat = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) 

        self.final_conv = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding=1) 

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def freeze_patch_emb(self):
        self.patch_embed.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}  # has pos_embed may be better

    def forward(self, x_low_reso, x_high_resolu):
        """
        x_low_reso: [B, 96, 256, 256]
        x_high_reso: [B, 96, 256, 256]
        return [B, 96, 256, 256]
        """
        x = torch.cat([x_low_reso, x_high_resolu], dim=1)
        h = self.first_conv(x)
        h = self.activation(h)
        h = self.second_conv(h)
        h = self.activation(h) # [B=2, C=128, H=256, W=256]
        
        h, H, W = self.patch_embed(h) # [B, N, C]

        for i in range(1, self.num_blocks+1):
            block_i = getattr(self, f'block{i}')
            h = block_i(h, H, H) # [B, N, C]

        h = h.permute(0, 2, 1) # [B, C, N=H*W]
        h = h.view(h.shape[0], h.shape[1], H, W) # [B=2, C=1024, H=256, W=256]
        h = self.pixel_shuffle(h)

        h = torch.cat([h, x_low_reso], dim=1) #  [B, 256+96, 256, 256]

        h = self.first_conv_after_cat(h)
        h = self.activation(h)
        h = self.second_conv_after_cat(h)
        h = self.activation(h)
        h = self.third_conv_after_cat(h)
        h = self.activation(h)

        out = self.final_conv(h)
        return out

