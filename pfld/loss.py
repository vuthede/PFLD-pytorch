import torch
from torch import nn
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize, using_wingloss=False):
        # Degree to radian
        euler_angle_gt = euler_angle_gt*3.14/180.0
        
        # print(f"Angle: {angle}. euler_angle_gt: {euler_angle_gt}")
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        attributes_w_n = attribute_gt[:, 1:6].float()
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        # print("---------------mat raito 1:", mat_ratio)
        mat_ratio = torch.Tensor([
            1.0 / (x+0.00001) if x > 0 else 0 for x in mat_ratio
        ]).to(device)
        # print("mat raito 2:", mat_ratio)

        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)

        weight_attribute = (weight_attribute<0.0001)*1 + weight_attribute 

        # print(f"weight_attribut: ", weight_attribute)
        if using_wingloss:
            l2_distant = customed_wing_loss(landmark_gt, landmarks)
        else:
            l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)

        return torch.mean(weight_angle * weight_attribute * l2_distant), torch.mean(l2_distant)


def customed_wing_loss(y_true, y_pred, w=10.0, epsilon=2.0):
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    x = y_true - y_pred
    absolute_x = torch.abs(x)

    losses = torch.where(w > absolute_x, w * torch.log(1.0 + absolute_x/epsilon), absolute_x - c)
    losses = torch.sum(losses, axis=1)

    return losses # Mean wingloss for each sample in batch


def smoothL1(y_true, y_pred, beta = 1):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    mae = torch.abs(y_true - y_pred)
    loss = torch.sum(torch.where(mae>beta, mae-0.5*beta , 0.5*mae**2/beta), axis=-1)
    return torch.mean(loss)

def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0, N_LANDMARK = 106):
    y_pred = y_pred.reshape(-1, N_LANDMARK, 2)
    y_true = y_true.reshape(-1, N_LANDMARK, 2) 
    
    x = y_true - y_pred
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(w > absolute_x, w * torch.log(1.0 + absolute_x/epsilon), absolute_x - c)
    loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0)
    return loss