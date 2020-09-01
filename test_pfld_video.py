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
from models.resnet import resnet101PFLD
from mtcnn.detector import MTCNN
# from mtcnn import MTCNN
from loss.LSE_loss import LandmarkDetectorAbstract, calculateLSEInOneVideo
import dlib
from imutils import face_utils
import utils
import hyperameter
import os
from pathlib import Path


cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LMK98_2_68_MAP = hyperameter.LMK98_2_68_MAP

class  LandmarkDetectorPFLD(LandmarkDetectorAbstract):
    def __init__(self, device, pfld_backbone, model_path):
        '''
        device:
        pfld_backbone: face detector backbone (code link to file pfld.py)
        model_path: where to save checkpoint of pfld_backbone
        '''
        # Face detector
        self.mtcnn = MTCNN()
        # self.plfd_backbone = CustomizedGhostNet(width=1, dropout=0.2).to(device)
        # self.plfd_backbone = PFLDInference()
        # self.plfd_backbone = AuxiliaryNet()
        # self.plfd_backbone = resnet101PFLD()
        self.plfd_backbone = pfld_backbone
        self.checkpoint = torch.load(model_path, map_location=device)
        self.plfd_backbone.load_state_dict(self.checkpoint['plfd_backbone'])
        self.plfd_backbone.eval()
        self.padding = 0
        self.first = True
        self.buffer_bbox = []
        self.tracker = cv2.TrackerKCF_create()

    def make_box_square(self, box):
        x,y, x1, y1 = box
        mx = (x+x1)//2
        my = (y+y1)//2
        pad = max(x1-x, y1-y)//2
        x = mx-pad
        y = my-pad
        x1 = mx+pad
        y1 = my+pad
        return [x, y, x1, y1]

    def make_box_square_v2(self, box, width, height):
        x1, y1, x2, y2 = np.array(box).astype(np.int32)

        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size = int(max([w, h])*1.1)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        return [x1, y1, x2, y2]

    def get_bbox(self, image):
        '''get face bounding box'''
        with torch.no_grad():
            box = self.mtcnn.detect_single_face(image,100) # x1, y1, x2, y2 Here is mtcnn in this repo
            # print('box: {}, self.first: {}'.format(box, self.first))
            # Create bbox tracker on face in image
            (height, width) = image.shape[:2]
            # print('len box and self.first: ', len(box), self.first)
            if (len(box)==4) & (self.first):
                box = self.make_box_square_v2(box, width, height)
                self.tracker.init(image, (box[0], box[1], box[2] - box[0], box[3] - box[1]))
                self.first = False
            else:
                ok, box = self.tracker.update(image)
                if np.sum(box) == 0: # tracker do not track face
                    # update bbox empty and enable first image
                    box = np.array([])
                    self.first = True
                    self.tracker = cv2.TrackerKCF_create()
                else:
                    box = list(box)
                    box[2] += box[0]
                    box[3] += box[1]
                    box = self.make_box_square_v2(box, width, height)
        return box

    def get_68_landmarks(self, image, isReturnBbox=True, isBuffer=True):
        landmarks = []
        # Whether or not do you take image from buffer bbox
        if (isBuffer is True) & (len(self.buffer_bbox) >= 1):
            box = self.buffer_bbox[-1]
        else:
            box = self.get_bbox(image)
        # Only use bbox with scale >= 10
        if min(box[2]-box[0], box[3]-box[1]) <= 10:
            if len(self.buffer_bbox) > 0:
                box = self.buffer_bbox[-1]
        else:
            # append bbox into buffer_bbox
            # print('bbox save: ', box)
            self.buffer_bbox.append(box)
        # Predict landmark for image
        x,y,x1,y1 = map(int, box)
        x = max(0, x-self.padding)
        y = max(0, y-self.padding)
        x1 = min(image.shape[1], x1+self.padding)
        y1 = min(image.shape[0], y1+self.padding)
        # Adjust bounding box
        adj_bbox = (x, y, x1, y1)
        # Crop face
        face = image[y:y1, x:x1]
        face = cv2.resize(face, (112, 112))
        face = transforms.Compose([transforms.ToTensor()])(face)
        face = torch.unsqueeze(face, 0)
        # Detect landmark
        _, landmarks = self.plfd_backbone(face)   
        landmarks = landmarks.detach().numpy()
        landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) # landmark 
        landmarks = np.squeeze(landmarks, 0)
        # Return landmark68
        indice68lmk = np.array(list(LMK98_2_68_MAP.keys()))
        landmarks = landmarks[indice68lmk] * np.array([112,112])
        # Convert landmarks coordinator to origin scale
        landmarks[:,0] = landmarks[:,0] * ((x1-x)/112) + x
        landmarks[:,1] = landmarks[:,1] * ((y1-y)/112) + y
        landmarks = landmarks.astype(int)
        assert len(landmarks)==68, f"There should be 68 landmark. Found {len(landmarks)}"
        # if self.out is not None:
        #     self.out.write(image)
        # cv2.imshow("Landmark predict: ", image)
        if isReturnBbox:
            return landmarks, adj_bbox
        else:
            return landmarks

    def get_68_landmarks_v2(self, image, isReturnBbox=True):
        '''
        detect landmarks and bbox for image
        '''
        landmarks = []
        box = self.get_bbox(image)
        if len(box):
            print('len(box) True : ', len(box))
            # Predict landmark for image
            x,y,x1,y1 = map(int, box)
            x = max(0, x-self.padding)
            y = max(0, y-self.padding)
            x1 = min(image.shape[1], x1+self.padding)
            y1 = min(image.shape[0], y1+self.padding)
            # Adjust bounding box
            adj_bbox = (x, y, x1, y1)
            # Crop face
            face = image[y:y1, x:x1]
            face = cv2.resize(face, (112, 112))
            face = transforms.Compose([transforms.ToTensor()])(face)
            face = torch.unsqueeze(face, 0)
            # Detect landmark
            _, landmarks = self.plfd_backbone(face)   
            landmarks = landmarks.detach().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) # landmark 
            landmarks = np.squeeze(landmarks, 0)
            # Return landmark68
            indice68lmk = np.array(list(LMK98_2_68_MAP.keys()))
            landmarks = landmarks[indice68lmk] * np.array([112,112])
            # Convert landmarks coordinator to origin scale
            landmarks[:,0] = landmarks[:,0] * ((x1-x)/112) + x
            landmarks[:,1] = landmarks[:,1] * ((y1-y)/112) + y
            landmarks = landmarks.astype(int)
            assert len(landmarks)==68, f"There should be 68 landmark. Found {len(landmarks)}"
            # if self.out is not None:
            #     self.out.write(image)
            # cv2.imshow("Landmark predict: ", image)
        else:
            print('len(box) false: ', len(box))
            landmarks = None
            adj_bbox = None
            self.first = True
            
        if isReturnBbox:
            return landmarks, adj_bbox
        else:
            return landmarks

    def draw_68_landmarks(self, image, isBuffer=True):
        landmarks, box = self.get_68_landmarks(image, isReturnBbox=True, isBuffer=isBuffer)
        # print(box, isBuffer)
        (x, y) = (box[0], box[1])
        (x1, y1) = (box[2], box[3])
        # Draw landmark on original image
        image = cv2.rectangle(image, (x,y), (x1,y1), (0,255,0),2)
        for l in landmarks:
            cv2.circle(image, (l[0], l[1]), 1, (255,0,0), 3)
        return image

    
    def draw_68_landmarks_v2(self, image):
        '''
        draw landmark and bbox for image
        '''
        landmarks, box = self.get_68_landmarks_v2(image, isReturnBbox=True)
        # print(box, isBuffer)
        if (landmarks is not None) & (box is not None):
            (x, y) = (box[0], box[1])
            (x1, y1) = (box[2], box[3])
            # Draw landmark on original image
            image = cv2.rectangle(image, (x,y), (x1,y1), (0,255,0),2)
            for l in landmarks:
                cv2.circle(image, (l[0], l[1]), 2, (255, 255, 255), 2)
        return image

class PFLDLandmarkDetector(LandmarkDetectorAbstract):
    '''
    this is landmark detector using pretrained-model on dlib library
    '''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./model.dat")
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
        for l in landmarks:
            cv2.circle(image, (l[0], l[1]), 2, (255,0,0), 3)
        cv2.imshow("Landmark predict: ", image)
        return landmarks


def main():
    # model_path = ["/home/vuthede/checkpoints_landmark/mobilenetv2/checkpoint_epoch_969.pth.tar"
    # "./checkpoints_landmark/mobilenetv2/checkpoint_epoch_230.pth.tar",
    # "./checkpoints_landmark/resnet101/checkpoint_epoch_403.pth.tar"]
    # video ["./data/dms/MiddleUpper_RGB.mp4",
    # "./data/dms/FrontWheel_IR.mp4",
    # "./data/dms/LeftUpper_RGB.mp4",
    # "./data/dms/LeftUpper_IR.mp4"]

    # Write video result output
    folder_path = "/".join(args["output"].split("/")[:-1])
    path = Path(folder_path)
    if not os.path.exists(folder_path):
        path.mkdir(parents=True)
    
    # Initialize face landmark backbone model
    pfld_backbone = None
    if args["model"] == "resnet101":
        pfld_backbone = resnet101PFLD()
    elif args["model"] == "ghostnet":
        pfld_backbone = CustomizedGhostNet(width=1, dropout=0.2)
    elif args["model"] == "auxilary":
        pfld_backbone = AuxiliaryNet()
    elif args["model"] == "mobilenetv2":
        pfld_backbone = PFLDInference()
    lmdetector = LandmarkDetectorPFLD(device=device, pfld_backbone=pfld_backbone, model_path=args["path_checkpoint"])
    
    # Capture video
    cap = cv2.VideoCapture(args["input"])
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_opt_path = args["output"]
    fwidth = 1920
    fheight = 1080
    out = cv2.VideoWriter(video_opt_path, cv2.VideoWriter_fourcc(*args["codec"]), args["fps"], (fwidth, fheight), True)
    
    i = 0
    while True:
        ret, img_org = cap.read()
        if img_org is not None:
            img = utils.rotate_frame(img_org, rotation=90)
            # (fheight, fwidth) = img.shape[:2]
        else:
            next
        # Up sacale image 200%
        # img = utils.rescale_frame(img, percent=200)
        if not ret:
            break
        # predict and draw landmark on original image
        # If i%2==0 then predict bbox image
        # if i%4==0:
        #     img = lmdetector.draw_68_landmarks(img, isBuffer=True)
        # else: # take bbox from buffer image
        #     img = lmdetector.draw_68_landmarks(img, isBuffer=False)
        img = lmdetector.draw_68_landmarks_v2(img)
        out.write(img)
        cv2.imshow("Result " + args["model"], img)

        # if the `q` key was pressed, break from the loop
        # if esc key was pressed, escape video streaming
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key==27:
            cv2.destroyAllWindows()
        i+=1
    out.release()

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
        help="path to input video file")
    ap.add_argument("-m", "--model", required=True,
        help="face landmark backbone model type")
    ap.add_argument("-o", "--output", required=True,
        help="Video output of facelandmark")
    ap.add_argument("-p", "--path_checkpoint", type=str, default="",
        help="Path save checkpoint of face landmark model")
    ap.add_argument("-f", "--fps", type=int, default=10,
        help="FPS of output video")
    ap.add_argument("-c", "--codec", type=str, default="MJPG",
        help="codec of output video")
    args = vars(ap.parse_args())
    main()

# #How to predict and save video? Run bellow command:
# python3 test_pfld_video.py -i ./video/input/Le_Trung_Ngoc_Huyen_19970122_Female/11.extreme.mp4 -o ./video/output/Le_Trung_Ngoc_Huyen_19970122_Female/mobilenetv2/11.extreme.avi -m mobilenetv2 -p ./checkpoints_landmark/mobilenetv2/checkpoint_epoch_230.pth.tar

