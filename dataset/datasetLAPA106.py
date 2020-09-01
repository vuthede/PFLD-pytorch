import numpy as np
import cv2
import sys
sys.path.append('..')

from torch.utils import data
import glob



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
    
    img, lmks = dataset[1]
    lmks = np.reshape(lmks, (-1,2))

    import matplotlib.pyplot as plt

    for p in lmks:
        p = p.astype(int)
        cv2.circle(img, tuple(p), 1, (255, 0, 0), 1)
    
    plt.imshow(img)
    plt.show()
