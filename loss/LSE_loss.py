"""
Implement Landmark Stability Loss mentioned in that 
https://www.researchgate.net/publication/332866914_Analysis_and_Improvement_of_Facial_Landmark_Detection
@author : De Vu
"""


import cv2
from abc import ABCMeta, abstractmethod, ABC
import glob
import cv2
import numpy as np
import dlib
from imutils import face_utils

class LandmarkDetectorAbstract(ABC):
    @abstractmethod
    def get_68_landmarks(image):
        """
        In here you have to implement everything you need to return 68 landmark coordimation
        given an image. Including face detector + landmark detector
        At the end of this method, we will get 2D numpy array with len==68
        """
        raise NotImplementedError("You have to implement this method. \
                                   Input is image, output are 2D numpy array representing coordination of landmarks ")


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
        return self.get_rect_and_keypoints(image)



def calculateLSEInOneVideo(lmdetector, videopath, annodir):
    def get_gt_landmark_from_file(anno):
        file1 = open(anno, 'r') 
        ls = file1.readlines() 
        ls = ls[3:-1] # Get onlu lines that contain landmarks. 68 lines

        lm = []
        for l in ls:
            l = l.replace("\n","")
            a = l.split(" ")
            a = [float(i) for i in a]
            lm.append(a)
        
        lm = np.array(lm)
        assert len(lm)==68, "There should be 68 landmarks. Get {len(lm)}"
        return lm

    anno_files = glob.glob(annodir + "/*.pts")
    cap = cv2.VideoCapture(videopath)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert len(anno_files) == num_frame, f"Number of annotation files {len(anno_files)} \
                                         is not equal to number of frames {num_frame} "
    
    lse_list = [] # List losses in all frames
    pre_gt_landmark = None
    pre_pred_landmark  = None
    for i, anno in enumerate(anno_files):
        ret, frame = cap.read()

    
        gt_landmark = get_gt_landmark_from_file(anno)
        pred_landmark = lmdetector.get_68_landmarks(frame)

        assert gt_landmark.shape == pred_landmark.shape, f"Shape of pred landmark is \
                                                            different from gt landmark {gt_landmark.shape}"
        
        # Calculate LSE for this frame
        N=68
        interocular = np.linalg.norm(gt_landmark[36] - gt_landmark[45])
        if i==0: # The first frame
            sum_delta = 0
        else:
            sum_delta = np.sum(np.linalg.norm((gt_landmark-pre_gt_landmark) - (pred_landmark-pre_pred_landmark), axis=1))
        lse_one_frame = sum_delta/(interocular*N)
        lse_list.append(lse_one_frame)

        print(f"LSE frame {i}: {lse_one_frame}")

        # Cache the precious predicted and gt landmark for later use in the next frame
        pre_gt_landmark = gt_landmark
        pre_pred_landmark = pred_landmark


    lse_video = sum(lse_list)/len(lse_list)

    return lse_video

if __name__=="__main__":
    pdld_lm_detector =  PFLDLandmarkDetector()
    video1 = "/hdd/data/VinAI/300VW_Dataset_2015_12_14/007/vid.avi"
    anno1 = "/hdd/data/VinAI/300VW_Dataset_2015_12_14/007/annot"
    lse = calculateLSEInOneVideo(pdld_lm_detector, videopath=video1, annodir=anno1)
    print("LSE error in video: ", lse)
