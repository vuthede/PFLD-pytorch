# ------------------------------------------------------------------------------
# Copyright (c) Zhichao Zhao
# Licensed under the MIT License.
# Created by Zhichao zhao(zhaozhichao4515@gmail.com)
# ------------------------------------------------------------------------------
import argparse
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.datasets import WLFWDatasets

from mtcnn.detector import MTCNN
from loss.LSE_loss import LandmarkDetectorAbstract, calculateLSEInOneVideo

import sys
sys.path.insert(0, "../../AdaptiveWingLoss")

from core import models
from utils.utils import fan_NME, show_landmarks, get_preds_fromhm


cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LMK98_2_68_MAP = {0: 0,
 2: 1,
 4: 2,
 6: 3,
 8: 4,
 10: 5,
 12: 6,
 14: 7,
 16: 8,
 18: 9,
 20: 10,
 22: 11,
 24: 12,
 26: 13,
 28: 14,
 30: 15,
 32: 16,
 33: 17,
 34: 18,
 35: 19,
 36: 20,
 37: 21,
 42: 22,
 43: 23,
 44: 24,
 45: 25,
 46: 26,
 51: 27,
 52: 28,
 53: 29,
 54: 30,
 55: 31,
 56: 32,
 57: 33,
 58: 34,
 59: 35,
 60: 36,
 61: 37,
 63: 38,
 64: 39,
 65: 40,
 67: 41,
 68: 42,
 69: 43,
 71: 44,
 72: 45,
 73: 46,
 75: 47,
 76: 48,
 77: 49,
 78: 50,
 79: 51,
 80: 52,
 81: 53,
 82: 54,
 83: 55,
 84: 56,
 85: 57,
 86: 58,
 87: 59,
 88: 60,
 89: 61,
 90: 62,
 91: 63,
 92: 64,
 93: 65,
 94: 66,
 95: 67}


out = cv2.VideoWriter('/home/vuthede/Desktop/adaptive_wingloss_box_extended.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (600,400))


class  LandmarkDetectorAdaptiveWingloss(LandmarkDetectorAbstract):
    def __init__(self, device, model_path):
        self.mtcnn = MTCNN()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.model = self.init_adaptive_wingloss_model(model_path)



    def init_adaptive_wingloss_model(self, model_path):
        PRETRAINED_WEIGHTS = model_path
        HG_BLOCKS = 4
        END_RELU = False 
        GRAY_SCALE = False
        NUM_LANDMARKS = 98
        model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)

        if PRETRAINED_WEIGHTS != "None":
            checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=torch.device('cpu'))    


        if 'state_dict' not in checkpoint:
            model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = model_ft.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                if k in model_weights}
            model_weights.update(pretrained_weights)
            model_ft.load_state_dict(model_weights)
            print("Loaded weighted")

        model_ft = model_ft.to(device)

        return model_ft
       

    def get_68_landmarks(self, image):
        with torch.no_grad():
            box = self.mtcnn.detect_single_face(image) # x1, y1, x2, y2
            x,y,x1,y1 = map(int, box)
            padding=40
            x = max(0, x-padding)
            y = max(0, y-padding)
            x1 = min(image.shape[1], x1+padding)
            y1 = min(image.shape[0], y1+padding)
            face = image[y:y1, x:x1]

            face = cv2.resize(face, (256, 256))
            image_tensor = torch.FloatTensor(face).permute((2,0,1)).unsqueeze(0)
            outputs, boundary_channels = self.model(image_tensor)
           
            pred_heatmap = outputs[-1][:, :-1, :, :][0].detach().cpu()
            pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0))
            pred_landmarks = pred_landmarks.squeeze().numpy()
        
            landmarks = pred_landmarks*4
            print(landmarks)

            indice68lmk = np.array(list(LMK98_2_68_MAP.keys()))
            landmarks = landmarks[indice68lmk] 
            landmarks[:,0] = landmarks[:,0] * ((x1-x)/256) + x
            landmarks[:,1] = landmarks[:,1] * ((y1-y)/256) + y

            landmarks = landmarks.astype(int)

            image = cv2.rectangle(image, (x,y), (x1,y1), (0,255,0),2)
            for l in landmarks:
                cv2.circle(image, (l[0], l[1]), 1, (255,0,0), 10)
                
            image=cv2.resize(image, (600,400))
            out.write(image)
            
            cv2.imshow("Landmark predict: ", image)

            k = cv2.waitKey(1)

            assert len(landmarks)==68, f"There should be 68 landmark. Found {len(landmarks)}"

            return landmarks








  




def main(args):
    lmdetector = LandmarkDetectorAdaptiveWingloss(device="cpu", model_path=args.model_path)
    
    mlse = calculateLSEInOneVideo(lmdetector, args.video, args.annot)

    print("Average mlse: ", mlse)
    out.release()

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default="./checkpoint/snapshot/checkpoint.pth.tar", type=str)
    parser.add_argument('--video', required=True, type=str)
    parser.add_argument('--annot', required=True, type=str)
    parser.add_argument('--show_image', default=False, type=bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
