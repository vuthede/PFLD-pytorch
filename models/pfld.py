#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# pfld.py -
# written by  zhaozhichao
#
######################################################

import torch
import torch.nn as nn
import math
import sys

sys.path.insert(0, "./models")

from ghostnet import _make_divisible, GhostBottleneck, ConvBnAct
from mobilefacenet import ConvBlock, Bottleneck


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        # expand_ratio=1
        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self, alpha=1.0):
        super(PFLDInference, self).__init__()

        self.inplane = 64 #1x
        self.alpha  = alpha
        self.base_channel = int(self.inplane*self.alpha)

        self.conv1 = nn.Conv2d(
            3,  self.base_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.base_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
             self.base_channel,  self.base_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d( self.base_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual( self.base_channel,  self.base_channel, 2, False, 2)

        self.block3_2 = InvertedResidual( self.base_channel,  self.base_channel, 1, True, 2)
        self.block3_3 = InvertedResidual( self.base_channel,  self.base_channel, 1, True, 2)
        self.block3_4 = InvertedResidual( self.base_channel,  self.base_channel, 1, True, 2)
        self.block3_5 = InvertedResidual( self.base_channel,  self.base_channel, 1, True, 2)

        self.conv4_1 = InvertedResidual(self.base_channel, self.base_channel*2, 2, False, 2)

        self.conv5_1 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, False, 4)
        self.block5_2 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, True, 4)
        self.block5_3 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, True, 4)
        self.block5_4 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, True, 4)
        self.block5_5 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, True, 4)
        self.block5_6 = InvertedResidual(self.base_channel*2, self.base_channel*2, 1, True, 4)
        self.conv6_1 = InvertedResidual(self.base_channel*2, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)

        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))

        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return out1, landmarks


class CustomizedGhostNet(nn.Module):
    """
    This is a customized module using GhostNet instead of Mobilenet2 
    for inference
    """

    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 2]],
        # stage2
        [[3,  48,  24, 0, 1]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 1]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 1]], #The original number of channels here is 80, but I change to 64 so that it fit to the AuxiliaryNet 
        [[3, 200,  80, 0, 2],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 1]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ],

        # final
        # [[5, 16, 16, 0.25, 1]]
    ]



    def __init__(self, width=1.0, dropout=0.2):
        super(CustomizedGhostNet, self).__init__()
        # setting of inverted residual blocks
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        first_6_stages = []  # This one used for another branch
        remaining_stages = []
        block = GhostBottleneck
        for i, cfg in enumerate(self.cfgs):
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                            se_ratio=se_ratio))
                input_channel = output_channel

            if i<=5:
                first_6_stages.append(nn.Sequential(*layers))
            else:
                remaining_stages.append(nn.Sequential(*layers))
                

        # output_channel = _make_divisible(exp_size * width, 4)
        output_channel = 16
        remaining_stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        
        self.begining_blocks = nn.Sequential(*first_6_stages)
        self.remaining_blocks = nn.Sequential(*remaining_stages)  # 16x14x14

        self.relu = nn.ReLU(inplace=True)

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)


    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        features_for_auxiliarynet = self.begining_blocks(x)
        x = self.remaining_blocks(features_for_auxiliarynet)

        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)

        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))

        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)
    
        return features_for_auxiliarynet, landmarks


class CustomizedGhostNet2(nn.Module):
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1],
         [3,  48,  24, 0, 2], #56x56
        ],

        # stage 2
        [[3,  72,  24, 0, 1],
         [5,  72,  40, 0.25, 2] # 28x28
        ],

        # stage 3
        [[5, 120,  40, 0.25, 1],
         [3, 240,  80, 0, 2]  #14x14
        ],

        # stage 4
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1],
         [5, 672, 160, 0.25, 1]
        ],

        # stage5
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]

    def __init__(self, width=1.0, dropout=0.2):
        super(CustomizedGhostNet2, self).__init__()
        # setting of inverted residual blocks
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        first_2_stages = []  # This one used for another branch
        remaining_stages = []
        block = GhostBottleneck
        for i, cfg in enumerate(self.cfgs):
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                            se_ratio=se_ratio))
                input_channel = output_channel

            if i<=1:
                first_2_stages.append(nn.Sequential(*layers))
            else:
                remaining_stages.append(nn.Sequential(*layers))
                

        output_channel = _make_divisible(exp_size * width, 4)
        # output_channel = 16
        print(f"Input channel: {input_channel}. Output channel: {output_channel}")
        remaining_stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        
        self.begining_blocks = nn.Sequential(*first_2_stages)
        self.remaining_blocks = nn.Sequential(*remaining_stages)  # 960x14x14
        self.conv6 = ConvBnAct(output_channel, 16, 1)

        self.relu = nn.ReLU(inplace=True)

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)



    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        features_for_auxiliarynet = self.begining_blocks(x)  # 40x28x28
        x = self.remaining_blocks(features_for_auxiliarynet) # 960x14x14

        x = self.conv6(x)  # 16x14x14

        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)   # # 32x7x7
        x2 = self.avg_pool2(x)

        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x)) # [128, 1, 1]

        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)
    
        return features_for_auxiliarynet, landmarks


class MobileFacenet(nn.Module):
    bottleneck_setting = [
        # t, c , n ,s
        [2, 64, 5, 2],
        [4, 128, 1, 2],
        [2, 128, 6, 1],
        [4, 128, 1, 2],
        [2, 128, 2, 1]
    ]

    def __init__(self):
        super(MobileFacenet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.block_first = self._make_layer(block, [self.bottleneck_setting[0]])
        self.blocks_remain = self._make_layer(block, self.bottleneck_setting[1:])


        self.conv2 = ConvBlock(128, 512, 1, 1, 0)

        self.linear7 = ConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc = nn.Linear(256, 196)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        share_features = self.block_first(x)
        x = self.blocks_remain(share_features)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return share_features, x

class AuxiliaryNet(nn.Module):
    def __init__(self, alpha, base_channel=None):
        super(AuxiliaryNet, self).__init__()
        self.base_channel = int(64*alpha)
        if base_channel != None:
            self.base_channel = base_channel
        self.conv1 = conv_bn(self.base_channel, 128, 3, 2)  # Original of PFLd is 64 but I used 80  or 40 here to match with ghostnet/ghostnet2 model
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    import time
    import numpy as np
    from torchsummary import summary

    input = torch.randn(1, 3, 112, 112)
    plfd_backbone = PFLDInference(alpha=1)
    summary(plfd_backbone, (3,112,112))

    plfd_backbone.eval()
# #     auxiliarynet = AuxiliaryNet()
    times = []
    for i in range(100):
        t1 =  time.time()
        features, landmarks = plfd_backbone(input)
        t2 = time.time()-t1
        times.append(t2)
        # print("Time: ", t2)
    
    print("time average: ", np.mean(times))
#     angle = auxiliarynet(features)

#     print("angle.shape:{0:}, landmarks.shape: {1:}".format(
#         angle.shape, landmarks.shape))
