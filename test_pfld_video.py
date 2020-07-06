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

from models.pfld import PFLDInference, AuxiliaryNet, CustomizedGhostNet
from mtcnn.detector import MTCNN
# from mtcnn import MTCNN
from loss.LSE_loss import LandmarkDetectorAbstract, calculateLSEInOneVideo
import dlib
from imutils import face_utils


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


out = cv2.VideoWriter('/home/vuthede/Desktop/Pha_video_best_mobilenet_with_no_track.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (432,768))


class  LandmarkDetectorPFLD(LandmarkDetectorAbstract):
    def __init__(self, device, model_path):
        self.mtcnn = MTCNN()
        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.plfd_backbone = CustomizedGhostNet(width=1, dropout=0.2).to(device)
        self.plfd_backbone = PFLDInference()
        checkpoint = torch.load(model_path, map_location=device)
        self.plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        self.plfd_backbone.eval()
        self.first=True
        self.tracker = cv2.TrackerKCF_create()


    def get_68_landmarks(self, image):
        with torch.no_grad():
            # if self.first:
            #     box = self.mtcnn.detect_single_face(image) # x1, y1, x2, y2 Here is mtcnn in this repo
            #     self.tracker.init(image, (box[0]-20, box[1]-20, box[2] - box[0]+20, box[3] - box[1]+20))
            #     self.first = False
            # else:
            #     ok, box = self.tracker.update(image)
            #     box = list(box)
            #     box[2] += box[0]
            #     box[3] += box[1]
            box = self.mtcnn.detect_single_face(image) # x1, y1, x2, y2 Here is mtcnn in this repo
            if len(box):
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(image, (box[0], box[1], box[2] - box[0], box[3] - box[1]))
            else:
                ok, box = self.tracker.update(image)
                box = list(box)
                box[2] += box[0]
                box[3] += box[1]

            landmarks = []
            if len(box):
                padding=20
                x,y,x1,y1 = map(int, box)
                x = max(0, x-padding)
                y = max(0, y-padding)
                x1 = min(image.shape[1], x1+padding)
                y1 = min(image.shape[0], y1+padding)
                face = image[y:y1, x:x1]
                face = cv2.resize(face, (112, 112))
                face = self.transform(face)
                face = torch.unsqueeze(face, 0)
                _, landmarks = self.plfd_backbone(face)   
                landmarks = landmarks.cpu().numpy()
                landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) # landmark 
                landmarks = np.squeeze(landmarks, 0)
                indice68lmk = np.array(list(LMK98_2_68_MAP.keys()))
                landmarks = landmarks[indice68lmk] * np.array([112,112])
                landmarks[:,0] = landmarks[:,0] * ((x1-x)/112) + x
                landmarks[:,1] = landmarks[:,1] * ((y1-y)/112) + y

                landmarks = landmarks.astype(int)

                image = cv2.rectangle(image, (x,y), (x1,y1), (0,255,0),2)
                for l in landmarks:
                    cv2.circle(image, (l[0], l[1]), 1, (255,0,0), 3)
                
                assert len(landmarks)==68, f"There should be 68 landmark. Found {len(landmarks)}"
                
            # image=cv2.resize(image, (600,400))
            # out.write(image)
            print("Image shape: ", image.shape)
            out.write(image)
            cv2.imshow("Landmark predict: ", image)

            k = cv2.waitKey(0)

            

            return landmarks


class PFLDLandmarkDetector(LandmarkDetectorAbstract):
    

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/vuthede/VinAI/mydeformation/model.dat")

    def get_rect_and_keypoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        kps_list = []
        for rect in rects:
            kps = self.predictor(gray, rect)
            kps = face_utils.shape_to_np(kps)
            kps_list.append(kps)
        
        if len(kps_list):
            return kps_list[0]
    
        return []

    # Override
    def get_68_landmarks(self, image):
        landmarks =  self.get_rect_and_keypoints(image)

        # image = cv2.rectangle(image, (x,y), (x1,y1), (0,255,0),2)
        for l in landmarks:
            cv2.circle(image, (l[0], l[1]), 2, (255,0,0), 3)
            
        # image=cv2.resize(image, (600,400))
        # print("Image shape: ", image.shape)
        # out.write(image)

        print(image.shape)

        
        cv2.imshow("Landmark predict: ", image)

        k = cv2.waitKey(1)

        return landmarks

  




def main():
    # model_path = "/home/vuthede/checkpoints_landmark/mobilenetv2/checkpoint_epoch_969.pth.tar"
    model_path = "./checkpoint/snapshot/checkpoint.pth.tar"

    video = "/home/vuthede/Downloads/VID_20200630_221121_970.mp4"
    lmdetector = LandmarkDetectorPFLD(device="cpu", model_path=model_path)
    # lmdetector = PFLDLandmarkDetector()
    cap = cv2.VideoCapture(video)

    while True:
        ret, img = cap.read()
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        if not ret:
            break

        # cv2.imshow("Image", img)
        k = cv2.waitKey(1)


        lmdetector.get_68_landmarks(img)

    out.release()

if __name__=="__main__":
    main()


