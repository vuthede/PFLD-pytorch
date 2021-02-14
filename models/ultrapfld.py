#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import torch
from torch.nn import Module, AvgPool2d, Linear, Sequential, Conv2d, BatchNorm2d, ReLU
import torch.nn.functional as F
import math



def Conv_Block(in_channel, out_channel, kernel_size, stride, padding, group=1, has_bn=True, is_linear=False):
    return Sequential(
        Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding, groups=group, bias=False),
        BatchNorm2d(out_channel) if has_bn else Sequential(),
        ReLU(inplace=True) if not is_linear else Sequential()
    )


class InvertedResidual(Module):
    def __init__(self, in_channel, out_channel, stride, use_res_connect, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        exp_channel = in_channel * expand_ratio
        self.use_res_connect = use_res_connect
        self.inv_res = Sequential(
            Conv_Block(in_channel=in_channel, out_channel=exp_channel, kernel_size=1, stride=1, padding=0),
            Conv_Block(in_channel=exp_channel, out_channel=exp_channel, kernel_size=3, stride=stride, padding=1, group=exp_channel),
            Conv_Block(in_channel=exp_channel, out_channel=out_channel, kernel_size=1, stride=1, padding=0, is_linear=True)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_res(x)
        else:
            return self.inv_res(x)


class GhostModule(Module):
    def __init__(self, in_channel, out_channel, is_linear=False):
        super(GhostModule, self).__init__()
        self.out_channel = out_channel
        init_channel = math.ceil(out_channel / 2)
        new_channel = init_channel

        self.primary_conv = Conv_Block(in_channel, init_channel, 1, 1, 0, is_linear=is_linear)
        self.cheap_operation = Conv_Block(init_channel, new_channel, 3, 1, 1, group=init_channel, is_linear=is_linear)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channel, :, :]


class GhostBottleneck(Module):
    def __init__(self, in_channel, hidden_channel, out_channel, stride):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.ghost_conv = Sequential(
            # GhostModule
            GhostModule(in_channel, hidden_channel, is_linear=False),
            # DepthwiseConv-linear
            Conv_Block(hidden_channel, hidden_channel, 3, stride, 1, group=hidden_channel, is_linear=True) if stride == 2 else Sequential(),
            # GhostModule-linear
            GhostModule(hidden_channel, out_channel, is_linear=True)
        )

        if stride == 1 and in_channel == out_channel:
            self.shortcut = Sequential()
        else:
            self.shortcut = Sequential(
                Conv_Block(in_channel, in_channel, 3, stride, 1, group=in_channel, is_linear=True),
                Conv_Block(in_channel, out_channel, 1, 1, 0, is_linear=True)
            )

    def forward(self, x):
        return self.ghost_conv(x) + self.shortcut(x)


class PFLD_Ultralight(Module):
    def __init__(self, width_factor=1, input_size=112, landmark_number=98):
        super(PFLD_Ultralight, self).__init__()

        self.conv1 = Conv_Block(3, int(64 * width_factor), 3, 2, 1)
        self.conv2 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1, group=int(64 * width_factor))

        self.conv3_1 = GhostBottleneck(int(64 * width_factor), int(128 * width_factor), int(80 * width_factor), stride=2)
        self.conv3_2 = GhostBottleneck(int(80 * width_factor), int(160 * width_factor), int(80 * width_factor), stride=1)
        self.conv3_3 = GhostBottleneck(int(80 * width_factor), int(160 * width_factor), int(80 * width_factor), stride=1)

        self.conv4_1 = GhostBottleneck(int(80 * width_factor), int(240 * width_factor), int(96 * width_factor), stride=2)
        self.conv4_2 = GhostBottleneck(int(96 * width_factor), int(288 * width_factor), int(96 * width_factor), stride=1)
        self.conv4_3 = GhostBottleneck(int(96 * width_factor), int(288 * width_factor), int(96 * width_factor), stride=1)

        self.conv5_1 = GhostBottleneck(int(96 * width_factor), int(384 * width_factor), int(144 * width_factor), stride=2)
        self.conv5_2 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)
        self.conv5_3 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)
        self.conv5_4 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)

        self.conv6 = GhostBottleneck(int(144 * width_factor), int(288 * width_factor), int(16 * width_factor), stride=1)
        self.conv7 = Conv_Block(int(16 * width_factor), int(32 * width_factor), 3, 1, 1)
        self.conv8 = Conv_Block(int(32 * width_factor), int(128 * width_factor), input_size // 16, 1, 0, has_bn=False)

        self.avg_pool1 = AvgPool2d(input_size // 2)
        self.avg_pool2 = AvgPool2d(input_size // 4)
        self.avg_pool3 = AvgPool2d(input_size // 8)
        self.avg_pool4 = AvgPool2d(input_size // 16)

        self.fc = Linear(int(512 * width_factor), landmark_number * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x3 = self.avg_pool3(x)
        x3 = x3.view(x3.size(0), -1)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x4 = self.avg_pool4(x)
        x4 = x4.view(x4.size(0), -1)

        x = self.conv6(x)
        x = self.conv7(x)
        x5 = self.conv8(x)
        x5 = x5.view(x5.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3, x4, x5], 1)
        landmarks = self.fc(multi_scale)

        return landmarks



if __name__=="__main__":
    import time
    import numpy as np
    from torchsummary import summary


    model = PFLD_Ultralight(width_factor=1, input_size=112, landmark_number=68)
    summary(model, (3,112,112))
    model.eval()
    x = torch.rand((1,3,112,112))

    times=[]
    for i in range(100):
        t1 = time.time()
        lmks = model(x)
        t2 = time.time()
        times.append(t2-t1)    
    
    print(f"Time average: {np.mean(times)}")
    # print(lmks.shape)
    
