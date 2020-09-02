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
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset.datasets import WLFWDatasets
from dataset.datasetLAPA106 import LAPA106DataSet
from pfld.loss import PFLDLossNoWeight
from pfld.utils import AverageMeter
import wandb
import logging
from models.pfld import PFLDInference, AuxiliaryNet, CustomizedGhostNet, MobileFacenet
from models.rexnet import ReXNetV1
from models.resnet import resnet101PFLD, resnet101BottleNeckPFLD, resnet101PFLD106lmks

wandb.init(project="Pratical Facial Landmark Detection on LAPA data")
# wandb.config.backbone = "MobileNet-v2"
wandb.config.width_model = 1
# wandb.config.pfld_backbone = "RexNetV1" # Or MobileNet2 Or RexNet
wandb.config.pfld_backbone = "Resnet101PFLD106lmks" # It is customized for PFLD  
# wandb.config.ghostnet_width = 1
# wandb.config.ghostnet_with_pretrained_weight_image_net = True
wandb.config.using_wingloss = False
wandb.config.train_img = "/home/ubuntu/LaPa/train/images"
wandb.config.train_lmks = "/home/ubuntu/LaPa/train/landmarks"
wandb.config.valid_img = "/home/ubuntu/LaPa/val/images"
wandb.config.valid_lmks = "/home/ubuntu/LaPa/val/landmarks"




CONSOLE_FORMAT = "%(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"
logger = logging.getLogger(__name__)
logging.basicConfig(format=CONSOLE_FORMAT, level=logging.INFO)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("YOOOOOOOOOOOO Device: ", device)

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')




def train(train_loader, plfd_backbone, auxiliarynet, criterion, optimizer,
          epoch):
    losses = AverageMeter()
    num_batch = len(train_loader)
    i = 0
    for img, landmark_gt in train_loader:
        i += 1
        img = img.to(device)
        # attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        # auxiliarynet = auxiliarynet.to(device)
        plfd_backbone = plfd_backbone.to(device)
        features, landmarks = plfd_backbone(img)

        print(f'Shape landmark_gt :{landmark_gt.shape}. Landmarks :{landmarks.shape}')

        # angle = auxiliarynet(features)
        using_wingloss = wandb.config.using_wingloss
        if epoch<25:
            using_wingloss=False
        weighted_loss, loss = criterion(landmark_gt, landmarks, args.train_batchsize, using_wingloss=using_wingloss)
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        losses.update(loss.item())
        wandb.log({"metric/loss":loss.item()})
        wandb.log({"metric/weighted_loss": weighted_loss.detach().cpu().numpy()})
        logger.info(f"Epoch:{epoch}. Lr:{optimizer.param_groups[0]['lr']} Batch {i} / {num_batch} batches. Loss: {loss.item()}. Weighted_loss:{ weighted_loss.detach().cpu().numpy()}")
        
    return weighted_loss, loss


def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    preds = preds.reshape(preds.shape[0], -1, 2).cpu().numpy() # landmark 
    target = target.reshape(target.shape[0], -1, 2).cpu().numpy() # landmark_gt

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
       
        if L == 19:  # aflw
            interocular = 34 # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L==106:
            interocular = np.linalg.norm(pts_gt[67, ] - pts_gt[80, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular)

    return rmse


def validate(val_dataloader, plfd_backbone, auxiliarynet, criterion):
    plfd_backbone.eval()
    auxiliarynet.eval() 
    losses = []
    nme_list = []
    with torch.no_grad():
        for img, landmark_gt in val_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            plfd_backbone = plfd_backbone.to(device)
            _, landmark = plfd_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())
            nme_batch = compute_nme(landmark, landmark_gt)
            nme_list += list(nme_batch) # Concat list
    
    wandb.log({"metric/valid_nme_loss": np.mean(nme_list)})
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))
    print(f'NME loss :{np.mean(nme_list)}')

    return np.mean(losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler

    

    if wandb.config.pfld_backbone == "ResNet101":
        plfd_backbone = resnet101PFLD()
        base_channel_auxiliarynet = 128
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model, base_channel=base_channel_auxiliarynet).to(device)
        logger.info(f"Using ResNet101 backbone of PFLD backbone")
    
    elif wandb.config.pfld_backbone == "Resnet101PFLD106lmks":
        plfd_backbone = resnet101PFLD106lmks()
        base_channel_auxiliarynet = 128
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model, base_channel=base_channel_auxiliarynet).to(device)
        logger.info(f"Using ResNet101 backbone for 106 lmks")

    elif wandb.config.pfld_backbone == "ResNet101BotteNeck":
        plfd_backbone = resnet101BottleNeckPFLD()
        expansion=4 # Botteck of resnet have expansion=4 for the number of features as ouput
        base_channel_auxiliarynet = 128*4
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model, base_channel=base_channel_auxiliarynet).to(device)
        logger.info(f"Using ResNet101 with BotteNeck backbone of PFLD backbone")


    else:
        plfd_backbone = PFLDInference(alpha=wandb.config.width_model).to(device) # MobileNet2 defaut
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model).to(device)
        
        logger.info(f"Using MobileNet2 as backbone of PFLD backbone with width = {wandb.config.width_model}")

    


    # Load checkpoints
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        auxiliarynet.load_state_dict(checkpoint['auxiliarynet'])
        logger.info(f"Load weights {args.resume} into plfd and auxiliary")



    # Watch model by wandb
    wandb.watch(plfd_backbone)
    wandb.watch(auxiliarynet)

    criterion = PFLDLossNoWeight()
    optimizer = torch.optim.Adam(
        [{
            'params': plfd_backbone.parameters()
        }, {
            'params': auxiliarynet.parameters()
        }],
        lr=args.base_lr,
        weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience,factor=0.5, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200 ,gamma=0.1)


    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    lapadataset = LAPA106DataSet(wandb.config.train_img, wandb.config.train_lmks, transform)
    dataloader = DataLoader(
        lapadataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    lapa_val_dataset = LAPA106DataSet(wandb.config.valid_img, wandb.config.valid_lmks, transform)
    lapa_val_dataloader = DataLoader(
        lapa_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        wandb.log({'metric/learing_rate': optimizer.param_groups[0]['lr']})
        weighted_train_loss, train_loss = train(dataloader, plfd_backbone, auxiliarynet,
                                      criterion, optimizer, epoch)
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint({
            'epoch': epoch,
            'plfd_backbone': plfd_backbone.state_dict(),
            'auxiliarynet': auxiliarynet.state_dict()
        }, filename)

        val_loss = validate(lapa_val_dataloader, plfd_backbone, auxiliarynet,
                            criterion)
        
        wandb.log({"metric/val_loss": val_loss})

        #scheduler.step(val_loss)
        scheduler.step()
        writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('data/loss', {'val loss': val_loss, 'train loss': train_loss}, epoch)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.001, type=float) # Origial is 0.0001 for  ReduceLROnPlateau. 
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=4, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=1000, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        '--snapshot',
        default='./checkpoint/snapshot/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./checkpoint/train.logs", type=str)
    parser.add_argument(
        '--tensorboard', default="./checkpoint/tensorboard", type=str)
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument(
        '--dataroot',
        default='./data/train_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default='./data/test_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default=128, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
