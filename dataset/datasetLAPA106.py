import numpy as np
import cv2
import sys
sys.path.append('..')

from torch.utils import data
import glob


def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                             M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_

class LAPA106DataSet(data.Dataset):
    TARGET_IMAGE_SIZE = (112, 112)
    def __init__(self, img_dir, anno_dir, transforms=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transforms = transforms

        self.img_path_list = glob.glob(img_dir + "/*.jpg")

        

    def _get_106_landmarks(self, path):
        file1 = open(path, 'r') 
        ls = file1.readlines() 
        ls = ls[1:] # Get only lines that contain landmarks. 68 lines

        lm = []
        for l in ls:
            l = l.replace("\n","")
            a = l.split(" ")
            a = [float(i) for i in a]
            lm.append(a)
        
        lm = np.array(lm)
        assert len(lm)==106, "There should be 106 landmarks. Get {len(lm)}"
        return lm
    
    def __getitem__(self, index):
        f = self.img_path_list[index]
        self.img = cv2.imread(f)
        self.landmark = self._get_106_landmarks(f.replace(self.img_dir, self.anno_dir).replace(".jpg", ".txt"))

        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        boxsize = int(np.max(wh)*1.2)
        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = self.img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = self.img[y1:y2, x1:x2]

        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        # Original cut face and lamks
        imgT_original = cv2.resize(imgT, self.TARGET_IMAGE_SIZE)
        landmark_original = (self.landmark - xy)/boxsize
        assert (landmark_original >= 0).all(), str(landmark_original) + str([dx, dy])
        assert (landmark_original <= 1).all(), str(landmark_original) + str([dx, dy])
        if self.transforms:
            imgT_original = self.transforms(imgT_original)

        # Random augmeentation
        fail_augment = False
        angle = np.random.randint(-30, 30)
        cx, cy = center
        cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
        cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
        M, landmark = rotate(angle, (cx,cy), self.landmark)

        imgT = cv2.warpAffine(self.img, M, (int(self.img.shape[1]*1.1), int(self.img.shape[0]*1.1)))

        
        wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
        size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
        xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
        landmark = (landmark - xy) / size
        if (landmark < 0).any() or (landmark > 1).any():
            fail_augment=True

        if fail_augment:
            return imgT_original, landmark_original

        x1, y1 = xy
        x2, y2 = xy + size
        height, width, _ = imgT.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = imgT[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx >0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        imgT = cv2.resize(imgT, self.TARGET_IMAGE_SIZE)


        if self.transforms:
            imgT = self.transforms(imgT)
        
        return imgT, landmark

    def __getitem1__(self, index):
        f = self.img_path_list[index]
        self.img = cv2.imread(f)
        h, w, _ = self.img.shape
        self.img = cv2.resize(self.img, self.TARGET_IMAGE_SIZE)
        self.landmark = self._get_106_landmarks(f.replace(self.img_dir, self.anno_dir).replace(".jpg", ".txt"))
        self.landmark[:,0] = self.landmark[:,0] * (self.TARGET_IMAGE_SIZE[1]/w) 
        self.landmark[:,1] = self.landmark[:,1] * (self.TARGET_IMAGE_SIZE[0]/h) 
        self.landmark = np.reshape(self.landmark, (-1))

        self.landmark = self.landmark/self.TARGET_IMAGE_SIZE[0]


        if self.transforms:
            self.img = self.transforms(self.img)

        
        return (self.img, self.landmark.astype(np.float32))
        

    def __len__(self):
        return len(self.img_path_list)



if __name__ == "__main__":
    dataset = LAPA106DataSet(img_dir="/home/vuthede/data/LaPa/train/images",
                            anno_dir="/home/vuthede/data/LaPa/train/landmarks")
    
    img, lmks = dataset[20]
    lmks = np.reshape(lmks, (-1,2)) *112

    import matplotlib.pyplot as plt

    for p in lmks:
        p = p.astype(int)
        cv2.circle(img, tuple(p), 1, (255, 0, 0), 1)
    
    plt.imshow(img)
    plt.show()
