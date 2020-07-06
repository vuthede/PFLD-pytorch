
from dataset.dataset300W import W300Dataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from models.pfld import PFLDInference, AuxiliaryNet, CustomizedGhostNet, MobileFacenet


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

def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)
    # print(f"Preds shape:", preds.shape)
    # print(f"Target shape:", target.shape)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        # pts_pred *=112
        # pts_gt *=112

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
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse

def validate(w300_val_dataloader, plfd_backbone):
    plfd_backbone.eval()

    nme_list = []
    cost_time = []
    i = 0
    with torch.no_grad():
        for img, landmark_gt, bbox in w300_val_dataloader:
            i +=1
            # if i==200:
            #     break

          
            landmark_gt = landmark_gt.to(device)

            start_time = time.time()

            x,y,x1,y1 = np.array(bbox)

            # Crop the face and inference
            face = img[:,:,y:y1, x:x1]
            # print(face.shape)
            face = torch.squeeze(face, 0).numpy().transpose((1,2,0))
            # print("face shape:", face.shape)
            face = cv2.resize(face, (112,112))
            face = face.transpose((2,0,1))
            face = torch.Tensor(face).unsqueeze(0)
            face = face.to(device)

            # print("face after ahpe: ", face.shape)
            t1 = time.time()

            _, landmarks = plfd_backbone(face)
            print("Time: ", time.time()-t1)

            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) # landmark 
            landmarks = np.squeeze(landmarks, 0)
            # Convert back to original landlandmark
           
            indice68lmk = np.array(list(LMK98_2_68_MAP.keys()))
            landmarks = landmarks[indice68lmk] * np.array([112,112])

            landmarks[:,0] = landmarks[:,0] * ((x1-x)/112) + x
            landmarks[:,1] = landmarks[:,1] * ((y1-y)/112) + y

            landmarks = np.expand_dims(landmarks, 0)

            cost_time.append(time.time() - start_time)

            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy() # landmark_gt

            if args.show_image:
                show_img = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
                show_img = (show_img * 255).astype(np.uint8)
                np.clip(show_img, 0, 255)

                pre_landmark = landmarks[0] 

                cv2.imwrite("xxx.jpg", show_img)
                img_clone = cv2.imread("xxx.jpg")

                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(img_clone, (x, y), 1, (255,0,0),-1)
                
                img_clone = cv2.resize(img_clone, None, fx=0.5, fy=0.5)
                cv2.imshow("xx.jpg", img_clone)
                cv2.waitKey(1)

            nme_temp = compute_nme(landmarks, landmark_gt)
            print("Nme: ", nme_temp)
            for item in nme_temp:
                nme_list.append(item)

        # nme
        print('nme: {:.4f}'.format(np.mean(nme_list)))
        print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))


def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)
    # plfd_backbone = PFLDInference().to(device)
    # plfd_backbone = CustomizedGhostNet(width=1, dropout=0.2).to(device)
    plfd_backbone = MobileFacenet()
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone = plfd_backbone.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    w300_val_dataset = W300Dataset(args.test_dataset, transform)
    w300_val_dataloader = DataLoader(w300_val_dataset, batch_size=1, shuffle=False, num_workers=0)

    validate(w300_val_dataloader, plfd_backbone)

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default="/home/vuthede/checkpoints_landmark/mobilefacenet/checkpoint_epoch_139.pth.tar", type=str)
    parser.add_argument('--test_dataset', default='/home/vuthede/AI/supervision-by-registration/cache_data/lists/300W/300w.test.full.GTB', type=str)
    parser.add_argument('--show_image', default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()
    main(args)
