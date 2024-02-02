# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import warnings
from einops import rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from mmcv.cnn import ConvModule

from utils.commons.hparams import hparams


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class HeadMLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

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

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class mit_b0(MixVisionTransformer): # 3.319M
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        self.load_state_dict(torch.load('checkpoints/pretrained_ckpts/mit_b0.pth'), strict=False)


class mit_b1(MixVisionTransformer): # 13.151M 
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        self.load_state_dict(torch.load('checkpoints/pretrained_ckpts/mit_b1.pth'), strict=False)



class mit_b2(MixVisionTransformer): # 24.196M
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        self.load_state_dict(torch.load('checkpoints/pretrained_ckpts/mit_b2.pth'), strict=False)



class mit_b3(MixVisionTransformer): # 44.072M
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        self.load_state_dict(torch.load('checkpoints/pretrained_ckpts/mit_b3.pth'), strict=False)


class mit_b4(MixVisionTransformer): # 60.843M
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        self.load_state_dict(torch.load('checkpoints/pretrained_ckpts/mit_b4.pth'), strict=False)


class mit_b5(MixVisionTransformer): # 81.443M
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        self.load_state_dict(torch.load('checkpoints/pretrained_ckpts/mit_b5.pth'), strict=False)
    

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, segformer_scale='b3'):
        super().__init__()
        self.segformer_scale = segformer_scale
        
        self.in_channels = [64, 128, 320, 512] if self.segformer_scale != 'b0' else [32, 64, 160, 256]
        self.feature_strides = [4, 8, 16, 32]
        self.in_index = [0, 1, 2, 3]
        self.input_transform='multiple_select'
        self.dropout = nn.Dropout2d(0.1)
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        embedding_dim = self.embedding_dim = 256
        self.linear_c4 = HeadMLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = HeadMLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = HeadMLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = HeadMLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        if dist.is_initialized():
            self.linear_fuse = ConvModule(
                in_channels=embedding_dim*4,
                out_channels=embedding_dim,
                kernel_size=1,
                norm_cfg=dict(type='SyncBN', requires_grad=True)
            )
        else:
            self.linear_fuse = ConvModule(
                in_channels=embedding_dim*4,
                out_channels=embedding_dim,
                kernel_size=1,
                norm_cfg=dict(type='BN', requires_grad=True)
            )

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
    
    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)

        return x


# from modules.hidenerf.models.networks_stylegan2 import Conv2dLayer
from modules.eg3ds.models.networks_stylegan2 import Conv2dLayer
class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, up=1, down=1):
        super(conv, self).__init__()
        self.conv = Conv2dLayer(num_in_layers, num_out_layers, kernel_size, activation='elu', up=up, down=down)
        self.bn = nn.InstanceNorm2d(
            num_out_layers, track_running_stats=False, affine=True
        )

    def forward(self, x):
        return self.bn(self.conv(x))


class SegFormerImg2PlaneBackbone(nn.Module):
    def __init__(self, mode='b3'):
        super().__init__()
        mode2cls = {
            'b0': mit_b0,
            'b1': mit_b1,
            'b2': mit_b2,
            'b3': mit_b3,
            'b4': mit_b4,
            'b5': mit_b5,
        }
        self.mode = mode
        self.mix_vit = mode2cls[mode]()
        self.fuse_head = SegFormerHead(mode)

        self.to_plane_cnn = nn.Sequential(*[
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2.),
            nn.Conv2d(in_channels=256, out_channels=96, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x):
        """
        x: [B, 3, H=512, W=512]
        return:
            plane: [B, 96, H=256, W=256]
        """

        feats = self.mix_vit(x)
        fused_feat = self.fuse_head(feats)

        planes = self.to_plane_cnn(fused_feat)

        planes = planes.view(len(planes), 3, -1, planes.shape[-2], planes.shape[-1])
        planes_xy = planes[:,0]
        planes_xy = torch.flip(planes_xy, [2])
        planes_xz = planes[:,1]
        planes_xz = torch.flip(planes_xz, [2])
        planes_zy = planes[:,2]
        planes_zy = torch.flip(planes_zy, [2, 3])
        planes = torch.stack([planes_xy, planes_xz, planes_zy], dim=1) # [N, 3, C, H, W]
        
        return planes


class TemporalAttNet(nn.Module):
    """
    Used to smooth the secc_plane with a window input
    """
    def __init__(self, in_dim=96, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.conv2d_layers = nn.Sequential(*[
            # [B, C=96, T, H=224, W=224] ==> [B, 64, T, 112, 112]
            nn.Conv3d(in_dim, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.LeakyReLU(0.02, True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.LeakyReLU(0.02, True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1,2,2), count_include_pad=False),
            # [B, C=64, T, H=112, W=112] ==> [B, 32, T, 56, 56]
            nn.Conv3d(64, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.LeakyReLU(0.02, True),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.LeakyReLU(0.02, True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1,2,2), count_include_pad=False),
            # [B, C=32, T, H=56, W=56] ==> [B, 16, T, 28, 28]
            nn.Conv3d(32, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.LeakyReLU(0.02, True),
            nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.LeakyReLU(0.02, True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1,2,2), count_include_pad=False),
        ])

        self.conv3d_layers = nn.Sequential(*[
            # [B, C=16, T, H=28, W=28] ==> [B, 8, T, 14, 14]
            nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1,2,2), count_include_pad=False),
            # [B, C=8, T, H=14, W=14] ==> [B, 8, T, 7, 7]
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1,2,2), count_include_pad=False),
            # [B, C=8, T, H=7, W=7] ==> [B, 4, T, 1, 1]
            nn.Conv3d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.Conv3d(4, 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1, count_include_pad=False),
        ])

        self.to_attention_weights = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        y: [B, T] attention weights
        out: [B, C, H, W]
        """
        b,c,t,h,w = x.shape
        y = F.interpolate(x, size=(t, 224, 224), mode='trilinear')
        y = self.conv2d_layers(y) # [B, 16, 5, 28, 28]
        y = self.conv3d_layers(y) # [B, 1, T, 1, 1]
        y = y.squeeze(1, 3, 4) # [B, T]
        assert y.ndim == 2
        y = y.reshape([b, 1, t, 1, 1])
        out = (y * x).sum(dim=2)
        return out


class SegFormerSECC2PlaneBackbone(nn.Module):
    def __init__(self, mode='b0', out_channels=96, pncc_cond_mode='cano_src_tgt'):
        super().__init__()
        mode2cls = {
            'b0': mit_b0,
            'b1': mit_b1,
            'b2': mit_b2,
            'b3': mit_b3,
            'b4': mit_b4,
            'b5': mit_b5,
        }
        self.mode = mode
        self.pncc_cond_mode = pncc_cond_mode
        in_dim = 9 if pncc_cond_mode == 'cano_src_tgt' else 6
        self.prenet = Conv2dLayer(in_dim, 3, 1)
        self.mix_vit = mode2cls[mode]()
        self.fuse_head = SegFormerHead(mode)
        self.to_plane_cnn = nn.Sequential(*[
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2.),
            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        ])
        # if hparams['use_motion_smo_net']:
            # self.motion_smo_win_size = hparams['motion_smo_win_size']
            # self.smo_net = TemporalAttNet(in_dim=out_channels, seq_len=hparams['motion_smo_win_size'])

    def forward(self, x):
        """
        x: [B, 3, H=512, W=512] or [B, 3, T, H, W]
        return:
            plane: [B, 96, H=256, W=256]
        """
        # if hparams['use_motion_smo_net']:
            # assert x.ndim == 5
            # x = rearrange(x, "n c t h w -> (n t) c h w", t=self.motion_smo_win_size)
        x = self.prenet(x)
        feats = self.mix_vit(x)
        fused_feat = self.fuse_head(feats)
        planes = self.to_plane_cnn(fused_feat)

        # if hparams['use_motion_smo_net']:
            # planes = rearrange(planes, "(n t) c h w -> n c t h w", t=self.motion_smo_win_size)
            # planes = self.smo_net(planes)

        planes = planes.view(len(planes), 3, -1, planes.shape[-2], planes.shape[-1])
        planes_xy = planes[:,0]
        planes_xy = torch.flip(planes_xy, [2])
        planes_xz = planes[:,1]
        planes_xz = torch.flip(planes_xz, [2])
        planes_zy = planes[:,2]
        planes_zy = torch.flip(planes_zy, [2, 3])
        planes = torch.stack([planes_xy, planes_xz, planes_zy], dim=1) # [N, 3, C, H, W]
        
        return planes


# from modules.hidenerf.new_modules.texture2plane_parser import Texture2PlaneParser
# class SegFormerTexture2PlaneBackbone(nn.Module):
#     def __init__(self, mode='b1'):
#         super().__init__()
#         mode2cls = {
#             'b0': mit_b0,
#             'b1': mit_b1,
#             'b2': mit_b2,
#             'b3': mit_b3,
#             'b4': mit_b4,
#             'b5': mit_b5,
#         }
#         self.mode = mode
#         self.prenet = Conv2dLayer(5, 3, 1)
#         self.tex2plane_parser = Texture2PlaneParser()
#         self.mix_vit = mode2cls[mode]()
#         self.fuse_head = SegFormerHead(mode)

#         if hparams.get("new_tex_mode", False) is True:
#             self.to_plane_cnn1 = nn.Sequential(*[
#                 nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
#                 nn.UpsamplingBilinear2d(scale_factor=2.),
#             ])
#             self.to_plane_cnn2 = nn.Sequential(*[
#                 nn.Conv2d(in_channels=256*3, out_channels=256, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
#                 nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
#                 nn.Conv2d(in_channels=256, out_channels=96, kernel_size=3, stride=1, padding=1)
#             ])
#         else:
#             self.to_plane_cnn = nn.Sequential(*[
#                 nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
#                 nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
#                 nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
#                 nn.UpsamplingBilinear2d(scale_factor=2.),
#                 nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1),
#             ])
        
#     def forward(self, x, idx_pixel_to_plane):
#         """
#         x: [B, 3, H=512, W=512]
#         return:
#             plane: [B, 96, H=256, W=256]
#         """
#         feats = self.mix_vit(x)
#         fused_feat = self.fuse_head(feats) # [B, 256, 128, 128]
#         if hparams.get("new_tex_mode", False) is True:
#             fused_feat = self.to_plane_cnn1(fused_feat) # [B, 96, 256, 256]
#             fused_feat = fused_feat.unsqueeze(1).repeat([1, 3, 1, 1, 1]) # [B, 3, 96, 256, 256]
#             tex_plane = self.tex2plane_parser(fused_feat, idx_pixel_to_plane) # [B, 3, 96, 256, 256]
#             tex_plane = rearrange(tex_plane, "n k c h w -> n (k c) h w") # [B, 3*96, 256, 256]
#             tex_plane = self.to_plane_cnn2(tex_plane) # [B, 96, 256, 256]
#             tex_plane = rearrange(tex_plane, "n (k c) h w -> n k c h w", k=3, c=32) # [B, 3*96, 256, 256]
#         else:
#             fused_feat = self.to_plane_cnn(fused_feat) # [B, 32, 256, 256]
#             fused_feat = fused_feat.unsqueeze(1).repeat([1, 3, 1, 1, 1]) # [B, 3, 32, 256, 256]
#             tex_plane = self.tex2plane_parser(fused_feat, idx_pixel_to_plane) # [B, 3, 32, 256, 256]
#         return tex_plane
    

if __name__ == '__main__':
    import tqdm
    img2plane = SegFormerTexture2PlaneBackbone()
    img2plane.cuda()
    x = torch.randn([4, 3, 512, 512]).cuda()
    idx = torch.randint(low=0, high=128*128, size=[4, 3, 256*256]).cuda()
    for _ in tqdm.trange(100):
        y = img2plane(x, idx)
    print(" ")