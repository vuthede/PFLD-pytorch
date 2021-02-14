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

# sys.path.insert(0, "./models")



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
            1,  self.base_channel, kernel_size=3, stride=2, padding=1, bias=False)
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

        # Like Auxilary net        
        self.conv3 = conv_bn(self.base_channel, 96, 3, 2)  # Original of PFLd is 64 but I used 80  or 40 here to match with ghostnet/ghostnet2 model
        self.conv4 = conv_bn(96, 128, 3, 2)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)


    def forward(self, x):  
        # print(x.shape)

        x = self.relu(self.bn1(self.conv1(x)))  
        # print(x.shape)

        x = self.relu(self.bn2(self.conv2(x)))  
        # print(x.shape)

        x = self.conv3_1(x)
        # print(x.shape)

        block3_2 = self.block3_2(x)
        # print(x.shape)

        block3_3 = self.block3_3(block3_2)
        # print(x.shape)

        block3_4 = self.block3_4(block3_3)
        # print(x.shape)

        block3_5 = self.block3_5(block3_4)

        conv3 = self.conv3(block3_5)

        # print(x.shape)

        conv4 = self.conv4(conv3)
        # print(x.shape)

        maxpool = self.max_pool1(conv4)
        # print(x.shape)

        x = maxpool.view(maxpool.size(0), -1)
        # print(x.shape)

        x = self.fc1(x)
        x = self.fc2(x)
        # print(x.shape)


        return x, (block3_2, block3_3, block3_4, block3_5, conv3, conv4, maxpool)



if __name__=="__main__":
    import time
    import numpy as np

    times = []

    pfld = PFLDInference()
    x = torch.rand((1,1,64,64))
    
    for i in range(100):
        t1 = time.time()
        pfld(x) 
        times.append(time.time()-t1)

    print("Time ave: ", np.mean(times))