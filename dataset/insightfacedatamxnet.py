import sys
import cv2

sys.path.insert(0, "/home/vuthede/VinAI/insightface/alignment/heatmapReg")
from data import FaceSegIter
import matplotlib.pyplot as plt

if __name__=="__main__":
    train_iter = FaceSegIter(path_imgrec ="/home/vuthede/data/bmvc_sdu_data2d/data_2d/train.rec",
      batch_size = 20,
      per_batch_size = 20,
      aug_level = 1,
      exf = 1,
      args = None,
      )

    
    img, hlabel, ann = train_iter.next_sample()

    for p in hlabel:
        cv2.circle(img, tuple(p), 2, (255, 0, 0), 2)
    
    plt.imshow(img)
    plt.show()
    print(f"Image shape: {img.shape}. type : {type(img)}")
    print(f"Halabel shape: {hlabel.shape}. Type :{type(hlabel)}")


