#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import logging
from pathlib import Path
import time
import os

import numpy as np
import torch

from torch.utils import data
from torch.utils.data import DataLoader

import wandb
import logging
from models.mobilenetforeyegaze import PFLDInference
from pfld.loss import PFLDEyeGazeLoss
from pfld.utils import AverageMeter
import cv2

import sys

## Deal with path
sys.path.insert(0, "GazeML/src")
from dataset.datasetEye import UnityEyeDataset


wandb.init(project="EyeGaze model")

CONSOLE_FORMAT = "%(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"
logger = logging.getLogger(__name__)
logging.basicConfig(format=CONSOLE_FORMAT, level=logging.INFO)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def train_one_epoch(traindataloader, model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    num_batch = len(traindataloader)
    i = 0
    for eye_img, gaze_ear in traindataloader:
        if i>50:
            break
        i += 1
        eye_img = eye_img.to(device)
        gaze_ear = gaze_ear.to(device)

        prediction = model(eye_img)

        l2_loss = criterion(gaze_ear, prediction)
        optimizer.zero_grad()
        l2_loss.backward()
        optimizer.step()

        losses.update(l2_loss.item())
        logger.info(f"Epoch:{epoch}. Lr:{optimizer.param_groups[0]['lr']} Batch {i} / {num_batch} batches. Loss: {l2_loss.item()}")

    return losses.avg
    


def draw_gaze(image_in, eye_pos, pitchyaw, length=15.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

def _put_text(img, text, point, color, thickness):
    img = cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, thickness, cv2.LINE_AA)
    return img

def draw_landmarks(img, lmks):
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, (255,0,0), -1, lineType=cv2.LINE_AA)

    return img


def vis_prediction_batch(batch, eye_imgs, gaze_ears, output="./vis"):
    """
    \eye_imgs batchx1x64x64
    \gaze_ears batchx3
    """
    if not os.path.isdir(output):
        os.makedirs(output)
    
    eye_imgs = eye_imgs.cpu().detach().numpy()
    gaze_ears = gaze_ears.cpu().detach().numpy()

    for i, (eye, gaze_ear) in enumerate(zip(eye_imgs, gaze_ears)):
        eye = (np.transpose(eye, (1,2,0))*255.0).astype(np.uint8)
        eye = draw_gaze(eye, (32, 32), gaze_ear[:2])
        eye = _put_text(eye, f'{gaze_ear[2]:02f}', (10,10), (0,255,0), 1)
        cv2.imwrite(f'{output}/Batch{batch}_sample{i}.png', eye)


def validate(valdataloader, model, criterion, epoch,optimizer, args):
    logFilepath  = os.path.join(args.snapshot, args.log_file)
    logFile  = open(logFilepath, 'a')

    model.eval()
    losses = AverageMeter()

    num_vis_batch = 40
    batch = 0
    with torch.no_grad():
        for eye_img, gaze_ear in valdataloader:
            if batch >40:
                break
            batch += 1
            eye_img = eye_img.to(device)
            gaze_ear = gaze_ear.to(device)

            prediction = model(eye_img)

            l2_loss = criterion(gaze_ear, prediction)
            losses.update(l2_loss.item())

            # Visualization
            vis_prediction_batch(batch, eye_img, prediction)
    
    logger.info(f"EValuation Epoch :{epoch}. Mean L2 loss: {losses.avg}")
    logFile.write(f"Epoch :{epoch}. Mean L2 loss: {losses.avg}. Lr:{optimizer.param_groups[0]['lr']}\n")

def main(args):
    # Init model
    model = PFLDInference()
    model.to(device)

    # Criterion
    criterion = PFLDEyeGazeLoss()

    # Train dataset, valid dataset
    train_dataset = UnityEyeDataset(data_dir=args.dataroot)
    val_dataset = UnityEyeDataset(data_dir=args.val_dataroot, augmentation=False)

    # Dataloader
    traindataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=2,
        drop_last=True)

    
    validdataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        [{
            'params': model.parameters()
        }],
        lr=0.001,
        weight_decay=1e-6)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience,factor=0.5, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30 ,gamma=0.8)
    
    for epoch in range(100000):
        train_one_epoch(traindataloader, model, criterion, optimizer, epoch)
        validate(validdataloader, model, criterion, epoch, optimizer,args)
        save_checkpoint({
            'epoch': epoch,
            'plfd_backbone': model.state_dict()
        }, filename=f'{args.snapshot}/epoch_{epoch}.pth.tar')
        scheduler.step()



def parse_args():
    parser = argparse.ArgumentParser(description='pfld')

    parser.add_argument(
        '--snapshot',
        default='./checkpoint/',
        type=str,
        metavar='PATH')

    parser.add_argument(
        '--log_file', default="log.txt", type=str)

    # --dataset
    parser.add_argument(
        '--dataroot',
        default='./data/train_data/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default='./data/test_data',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default=2, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

