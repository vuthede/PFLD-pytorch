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

from pfld.loss import PFLDLoss, PFLDLossNoWeight
from pfld.utils import AverageMeter
import wandb
import logging
from models.pfld import PFLDInference, AuxiliaryNet, CustomizedGhostNet, MobileFacenet
from models.rexnet import ReXNetV1
from models.resnet import resnet101PFLD, resnet101BottleNeckPFLD, resnet101PFLD68lmks
import mxnet as mx


wandb.init(project="Pratical Facial Landmark Detection Combine Stable Data")
# wandb.config.backbone = "MobileNet-v2"
wandb.config.width_model = 1
wandb.config.freeze_some_last_layers = False
# wandb.config.pfld_backbone = "RexNetV1"ResNet101BotteNeck # Or MobileNet2 Or RexNet
wandb.config.pfld_backbone = "resnet101PFLD68lmks" # It is customized for PFLD  
# wandb.config.ghostnet_width = 1
# wandb.config.ghostnet_with_pretrained_weight_image_net = True
wandb.config.using_wingloss = False
wandb.config.data_file_train_LP_rec = "/home/vuthede/data/pfld_1.0_68_style_LP_VW/pfld_train_data_LP.rec"
wandb.config.data_file_train_Style_rec = "/home/vuthede/data/pfld_1.0_68_style_LP_VW/pfld_train_data_style.rec"
wandb.config.data_file_train_VW_rec = "/home/vuthede/data/pfld_1.0_68_style_LP_VW/pfld_train_data_VW.rec"

wandb.config.data_file_valid_LP_rec = "/home/vuthede/data/pfld_1.0_68_style_LP_VW/pfld_test_data_LP.rec"
wandb.config.data_file_valid_Style_rec = "/home/vuthede/data/pfld_1.0_68_style_LP_VW/pfld_test_data_style.rec"
wandb.config.data_file_valid_VW_rec = "/home/vuthede/data/pfld_1.0_68_style_LP_VW/pfld_test_data_VW.rec"




CONSOLE_FORMAT = "%(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"
logger = logging.getLogger(__name__)
logging.basicConfig(format=CONSOLE_FORMAT, level=logging.INFO)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



def to_tensor(img, labels):
    img = torch.FloatTensor(img)
    labels = torch.FloatTensor(labels)
    return img, labels


def train(train_iter, plfd_backbone, auxiliarynet, criterion, optimizer,
          epoch, datasetname):

    losses = AverageMeter()
    num_batch = -1
    i = 0
    for batch in train_iter:    
        img   = batch.data[0].asnumpy()
        labels = batch.label[0].asnumpy()
        img, labels = to_tensor(img, labels)

        landmark_gt = labels[:, 0:68*2]

        i += 1
        img = img.to(device)
        landmark_gt = landmark_gt.to(device)
        plfd_backbone = plfd_backbone.to(device)
        # auxiliarynet = auxiliarynet.to(device)
        features, landmarks = plfd_backbone(img)


        using_wingloss = wandb.config.using_wingloss
        if epoch<25:
            using_wingloss=False
        weighted_loss, loss = criterion(landmark_gt,
                                       landmarks, args.train_batchsize, using_wingloss=using_wingloss)
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()  

        losses.update(loss.item())
        wandb.log({f"metric/loss_{datasetname}":loss.item()})
        wandb.log({f"metric/weighted_loss_{datasetname}": weighted_loss.detach().cpu().numpy()})
        logger.info(f"On {datasetname} dataset. Epoch:{epoch}. Lr:{optimizer.param_groups[0]['lr']} Batch {i} / {num_batch} batches. Loss: {loss.item()}. Weighted_loss:{ weighted_loss.detach().cpu().numpy()}")
        
        if i>=2:
            break

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
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular)

    return rmse




def validate(valid_iter, plfd_backbone, auxiliarynet, criterion, datasetname):
    plfd_backbone.eval()
    auxiliarynet.eval() 
    losses = []
    nme_list = []
    with torch.no_grad():
        for batch in valid_iter:    
            img   = batch.data[0].asnumpy()
            labels = batch.label[0].asnumpy()
            img, labels = to_tensor(img, labels)
            landmark_gt = labels[:, 0:68*2]

            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            plfd_backbone = plfd_backbone.to(device)
            _, landmark = plfd_backbone(img)


            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())
            nme_batch = compute_nme(landmark, landmark_gt)
            nme_list += list(nme_batch) # Concat list
            break
    
    wandb.log({f"metric/valid_nme_loss_{datasetname}": np.mean(nme_list)})
    print(f"===> Evaluate on {datasetname}")
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
    if wandb.config.pfld_backbone == "GhostNet":
        plfd_backbone = CustomizedGhostNet(width=wandb.config.ghostnet_width, dropout=0.2)
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model).to(device)

        logger.info(f"Using GHOSTNET with width={wandb.config.ghostnet_width} as backbone of PFLD backbone")

        # If using pretrained weight from ghostnet model trained on image net
        if (wandb.config.ghostnet_with_pretrained_weight_image_net == True):
            logger.info(f"Using pretrained weights of ghostnet model trained on image net data ")
            plfd_backbone = load_pretrained_weight_imagenet_for_ghostnet_backbone(
                plfd_backbone, "./checkpoint_imagenet/state_dict_93.98.pth")
            
    elif wandb.config.pfld_backbone == "MobileFaceNet":
        plfd_backbone = MobileFacenet()
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model).to(device)
        logger.info(f"Using MobileFacenetas backbone of PFLD backbone")
    
    elif wandb.config.pfld_backbone == "RexNetV1":
        plfd_backbone = ReXNetV1(width_mult=wandb.config.width_model)
        if wandb.config.width_model==1:
            base_channel_auxiliarynet = 38
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model, base_channel=base_channel_auxiliarynet).to(device)
        logger.info(f"Using ReXNetV1 backbone of PFLD backbone")

    elif wandb.config.pfld_backbone == "ResNet101":
        plfd_backbone = resnet101PFLD()
        base_channel_auxiliarynet = 128
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model, base_channel=base_channel_auxiliarynet).to(device)
        logger.info(f"Using ResNet101 backbone of PFLD backbone")

    elif wandb.config.pfld_backbone == "resnet101PFLD68lmks":
        plfd_backbone = resnet101PFLD68lmks()
        base_channel_auxiliarynet = 128
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model, base_channel=base_channel_auxiliarynet).to(device)
        logger.info(f"Using ResNet101 backbone of PFLD backbone for 68 landmarks")
    

    elif wandb.config.pfld_backbone == "ResNet101BotteNeck":
        plfd_backbone = resnet101BottleNeckPFLD()  
        writer.add_scalars('data/loss', {'val loss': val_loss, 'train loss': train_loss}, epoch)
        expansion=4 # Botteck of resnet have expansion=4 for the number of features as ouput
        base_channel_auxiliarynet = 128*4
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model, base_channel=base_channel_auxiliarynet).to(device)
        logger.info(f"Using ResNet101 with BotteNeck backbone of PFLD backbone")


    else:
        plfd_backbone = PFLDInference(alpha=wandb.config.width_model).to(device) # MobileNet2 defaut
        auxiliarynet = AuxiliaryNet(alpha=wandb.config.width_model).to(device)
        
        logger.info(f"Using MobileNet2 as backbone of PFLD backbone with width = {wandb.config.width_model}")

        if wandb.config.freeze_some_last_layers:
            plfd_backbone = load_weights_and_freeze_some_last_layers(plfd_backbone,
                                        mobilenetv2_checkpoint="./checkpoint_mobilenetv2/snapshot/checkpoint.pth.tar")
            logger.info(f"Using MobileNet2. Load weights from pretrained into some last layers and freeze them ")


    # Load checkpoints
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        auxiliarynet.load_state_dict(checkpoint['auxiliarynet'])
        logger.info(f"Load weights {args.resume} into plfd and auxiliary")

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


    ### Train and valid iters
    image_size=112
    batch_size=args.train_batchsize

    train_iter_LP = mx.io.ImageRecordIter(
        path_imgrec=wandb.config.data_file_train_LP_rec, 
        data_shape=(3, image_size, image_size), 
        batch_size=batch_size,
        label_width=139,
        shuffle = True,
        shuffle_chunk_size = 1024,
        seed = 1234, 
        prefetch_buffer = 10, 
        preprocess_threads = 2
    )

    train_iter_Style = mx.io.ImageRecordIter(
        path_imgrec=wandb.config.data_file_train_Style_rec, 
        data_shape=(3, image_size, image_size), 
        batch_size=batch_size,
        label_width=139,
        shuffle = True,
        shuffle_chunk_size = 1024,
        seed = 1234, 
        prefetch_buffer = 10, 
        preprocess_threads = 2
    )

    train_iter_VW = mx.io.ImageRecordIter(
        path_imgrec=wandb.config.data_file_train_VW_rec, 
        data_shape=(3, image_size, image_size), 
        batch_size=batch_size,
        label_width=139,
        shuffle = True,
        shuffle_chunk_size = 1024,
        seed = 1234, 
        prefetch_buffer = 10, 
        preprocess_threads = 2
    )


    valid_iter_LP = mx.io.ImageRecordIter(
        path_imgrec=wandb.config.data_file_valid_LP_rec, 
        data_shape=(3, image_size, image_size), 
        batch_size=batch_size,
        label_width=139,
        shuffle = True,
        shuffle_chunk_size = 1024,
        seed = 1234, 
        prefetch_buffer = 10, 
        preprocess_threads = 2
    )

    valid_iter_Style = mx.io.ImageRecordIter(
        path_imgrec=wandb.config.data_file_valid_Style_rec, 
        data_shape=(3, image_size, image_size), 
        batch_size=batch_size,
        label_width=139,
        shuffle = True,
        shuffle_chunk_size = 1024,
        seed = 1234, 
        prefetch_buffer = 10, 
        preprocess_threads = 2
    )

    valid_iter_VW = mx.io.ImageRecordIter(
        path_imgrec=wandb.config.data_file_valid_VW_rec, 
        data_shape=(3, image_size, image_size), 
        batch_size=batch_size,
        label_width=139,
        shuffle = True,
        shuffle_chunk_size = 1024,
        seed = 1234, 
        prefetch_buffer = 10, 
        preprocess_threads = 2
    )

    

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        wandb.log({'metric/learing_rate': optimizer.param_groups[0]['lr']})


        # Train different dataset
        weighted_loss_LP, loss_LP = train(train_iter_LP, plfd_backbone, auxiliarynet,
                                      criterion, optimizer, epoch, "LP")
        weighted_loss_style, loss_style = train(train_iter_Style, plfd_backbone, auxiliarynet,
                                      criterion, optimizer, epoch, "Style")
        
        weighted_loss_VW, loss_VW = train(train_iter_VW, plfd_backbone, auxiliarynet,
                                      criterion, optimizer, epoch, "VW")


        # filename = os.path.join(
        #     str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        

        val_loss_LP = validate(valid_iter_LP, plfd_backbone, auxiliarynet,
                            criterion, "LP")
        
        val_loss_Style = validate(valid_iter_Style, plfd_backbone, auxiliarynet,
                            criterion, "Style")
        
        val_loss_VW = validate(valid_iter_VW, plfd_backbone, auxiliarynet,
                            criterion, "VW")

        
        filename = f'{str(args.snapshot)}/checkpoint_epoch_{epoch}_lossLP_{val_loss_LP:.3f}_lossstyle_{val_loss_Style:.3f}_lossvw_{val_loss_VW:.3f}.pth.tar'

        save_checkpoint({
            'epoch': epoch,
            'plfd_backbone': plfd_backbone.state_dict(),
            'auxiliarynet': auxiliarynet.state_dict()
        }, filename)
        
        wandb.log({f"metric/mean_val_loss_LP": val_loss_LP,  f"metric/mean_val_loss_Style": val_loss_Style, f"metric/mean_val_loss_VW": val_loss_VW,
                  f"metric/mean_train_loss_LP": loss_LP,  f"metric/mean_train_loss_Style": loss_style, f"metric/mean_train_loss_VW": loss_VW})

        #scheduler.step(val_loss)
        scheduler.step()
      
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
    parser.add_argument('--train_batchsize', default=2, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
