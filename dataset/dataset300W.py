import numpy as np
import cv2
import sys
sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader


class W300Dataset(data.Dataset):
    def __init__(self, file_list, transforms=None):
        self.line = None
        self.landmarks = None
        self.bbox = None
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __get_68_landmark(self, path):
        file1 = open(path, 'r') 
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
    

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        # print("Line:", self.line)
        self.img = cv2.imread(self.line[0])
        # print(f"Shape iamge:", self.img.shape)
        self.landmark = self.__get_68_landmark(self.line[1])
        # print("Landmark shape: ", self.landmark.shape)

        self.bbox = [float(self.line[2]), float(self.line[3]), float(self.line[4]), float(self.line[5])]
        self.bbox = list(map(int, self.bbox))

        print(f"Bbox:", self.bbox)

        if self.transforms:
            self.img = self.transforms(self.img)
        
        
        return (self.img, self.landmark, self.bbox)

    def __len__(self):
        return len(self.lines)

if __name__=="__main__":
    dataset300W = W300Dataset("/home/vuthede/AI/supervision-by-registration/cache_data/lists/300W/300w.test.full.GTB")
    print("Len data:", len(dataset300W))
    dataset300W[0]
    
