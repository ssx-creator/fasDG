# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.backbones import ResNet
from mmdet.models.builder import SHARED_HEADS
from mmdet.models.utils import ResLayer as _ResLayer


@SHARED_HEADS.register_module()
class ResLayer(BaseModule):

    def __init__(self,
                 depth,
                 stage=3,
                 stride=2,
                 dilation=1,
                 style='pytorch',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 dcn=None,
                 pretrained=None,
                 init_cfg=None):
        super(ResLayer, self).__init__(init_cfg)

        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.stage = stage
        self.fp16_enabled = False
        block, stage_blocks = ResNet.arch_settings[depth]
        stage_block = stage_blocks[stage]
        planes = 64 * 2**stage
        inplanes = 64 * 2**(stage - 1) * block.expansion

        res_layer = _ResLayer(
            block,
            inplanes,
            planes,
            stage_block,
            stride=stride,
            dilation=dilation,
            style=style,
            with_cp=with_cp,
            norm_cfg=self.norm_cfg,
            dcn=dcn)
        self.add_module(f'layer{stage + 1}', res_layer)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

    @auto_fp16()
    def forward(self, x):
        res_layer = getattr(self, f'layer{self.stage + 1}')
        out = res_layer(x)
        return out

    def train(self, mode=True):
        super(ResLayer, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()




import torch
import torch.nn.functional as F
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x): #　1024 1024 14 14
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC 196 1024 1024
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC 197 1024 1024
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

from collections import OrderedDict
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


@SHARED_HEADS.register_module()
class ClipResLayer(BaseModule):

    def __init__(self,
                 stage=3,
                 layers=[3, 4, 23, 3],
                 norm_eval=False,
                 pretrained=None,
                 init_cfg=None):
        super(ClipResLayer, self).__init__(init_cfg)
        self.norm_eval = norm_eval
        self.stage = stage

        # layers=[3, 4, 23, 3]
        width=64
        # input_resolution=224
        # heads=32
        # output_dim=512
        self.layer4 = self._make_layer(1024, width * 8, layers[3], stride=2)
        # embed_dim = width * 32 # the ResNet feature dimension
        # self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def init_weights(self, pretrained=None, norm_freeze=False):
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location="cpu").float().state_dict()
            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            # w = w[300:]
            # print(u, 'are misaligned params in CLIPResLayer')
            print(w, 'are unexpexted params in CLIPResLayer')

            # free batchNorm
            if norm_freeze:
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        for param in m.parameters():
                            param.requires_grad = False

    def _make_layer(self, _inplanes, planes, blocks, stride=1):
        layers = [Bottleneck(_inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    @auto_fp16()
    def forward(self, x):
        x = self.layer4(x) #　1024 1024 14 14 
        # x = self.attnpool(x)
        return x

    def train(self, mode=True):
        super(ClipResLayer, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()