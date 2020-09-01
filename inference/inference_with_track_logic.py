from __future__ import division
from PIL import ImageFile
import PIL
import random
import torch
import time
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from mtcnn import MTCNN
import cv2
import math
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet import autograd
import os
import sys
import argparse
from mxnet.gluon.data.vision.transforms import Normalize
from mxnet.gluon import nn
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
import numbers
import insightface


lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
ImageFile.LOAD_TRUNCATED_IMAGES = True
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(
    sys.version_info)

def make_box_square(box):
    x,y, x1, y1 = box
    # Get mid point
    mx = (x+x1)//2
    my = (y+y1)//2
    # Padding from the middle
    pad = max(x1-x, y1-y)//2
    x = mx-pad
    y = my-pad
    x1 = mx + pad
    y1 = my+pad
    return [x, y, x1, y1]

def preprocess(image_int):
    #data = (data-123.0) / 58.0
    image_int = mx.nd.array(image_int).transpose([2, 0, 1])
    image_int = image_int.reshape(1, image_int.shape[0], image_int.shape[1], image_int.shape[2])
    image_float = image_int.astype('float32')/255
    # the following normalization statistics are taken from gluon model zoo
    normalizer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    image = normalizer(image_float)
    return image


"""
\return single face box (x1, t1, x2, y2) and its score
Using mtcnn
"""
def get_single_face(fd, rgb):
    # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    faces = fd.detect_faces(rgb)
    print(f'Len face :{len(faces)}')
    if len(faces):
        box =  faces[0]['box']
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        
        return faces[0]['box'], faces[0]['confidence']
    else:
        return None, None

def get_single_face_using_retina(fd, rgb):
    bboxes, scores = fd.detect(rgb)

    if len(bboxes):
        box = bboxes[0]
        # box[2] = box[0] + box[2]
        # box[3] = box[1] + box[3]

        return box, scores[0]
    else:
        return None, None 



def main(args):
    assert args.detector in ["mtcnn", "retina"]

    use_gpu = None
    devices = mx.gpu()

    # fd = MTCNN()

    ctx_id=0
    fd = insightface.model_zoo.get_model('retinaface_mnet025_v2')
    fd.prepare(ctx_id=ctx_id)
    # Define network
    net = gluon.nn.SymbolBlock.imports("../checkpoints_mxnet/lmks_detector-symbol.json", [
                                       'data'], "../checkpoints_mxnet/lmks_detector-1600.params", ctx=devices)
    cap = cv2.VideoCapture('/home/vuthede/data/car/FrontWheel_RGB.mp4')
    out = cv2.VideoWriter('/home/vinai/Desktop/' + args.model_type + '.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (int(480), int(640)))
    image = cv2.imread("/home/vinai/Desktop/IMG_2363.JPG")
    first = True
    prev = None 
    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()
        print("Image shape:", image.shape)
        if image is None:
            break
        # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ih, iw = image.shape[0], image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # bb, score = get_single_face(fd, image)
        bb, score = get_single_face_using_retina(fd, image)
        
        
        if len(bb):
            
            bb = list(map(int, bb))
            first = False
            tracker = cv2.TrackerKCF_create()
            tracker.init(image, (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]))
        else:
            ok, bb = tracker.update(image)
            print("Is track ok:", ok)
            bb = list(bb)
            bb[2] += bb[0]
            bb[3] += bb[1]
            # bb, score = fd.detect(image)
            # bb = list(map(int, bb[0]))
        h, w = (bb[3] - bb[1])*float(args.pre_crop_expand), (bb[2] -
                                                             bb[0])*float(args.pre_crop_expand)
        x1, y1 = int(max(math.floor(bb[0]-w), 0)
                     ), int(max(math.floor(bb[1]-h), 0))
        x2, y2 = int(min(math.ceil(bb[2]+w), iw)
                     ), int(min(math.ceil(bb[3]+h), ih))
        x1, y1, x2, y2 = make_box_square([x1, y1, x2, y2])
        print((x1, y1), (x2, y2))
        _face = image[y1:y2, x1:x2]
        face = cv2.resize(_face, (112, 112))
        face = preprocess(face)
        face = face.as_in_context(devices)
        shared_feature, regs = net(face)
        locations = nd.Flatten(regs).asnumpy().reshape(-1, 2) 
        # locations[:, 0] *= face.shape[-1]
        # locations[:, 1] *= face.shape[-2]
        scale_h, scale_w = _face.shape[0], _face.shape[1]
        locations[:, 0], locations[:, 1] = locations[:, 0] * \
            scale_w + x1, locations[:, 1] * scale_h + y1
        if prev is None:
            prev = locations
        else:
            locations = (locations + prev)/2.
            prev = locations
        for i in locations:
            cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", image)

        if cv2.waitKey(0)==27:
            break

        out.write(image)
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_crop_expand',  type=float, default=0.,
                        help='parameters for pre-crop expand ratio')
    parser.add_argument('--model_type',  type=str, default="output")
    parser.add_argument('--detector',  type=str, default="mtcnn")

    args = parser.parse_args()
    main(args)