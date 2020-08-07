import numpy as np
import cv2
import sys
sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader
import mxnet as mx




class LandmarkDatasetsCombined(data.Dataset):
    def __init__(self, data_file):
        """
        \data_file is the datafile have extension .rec
        """
        super(LandmarkDatasetsCombined, self).__init__()
        self.data_file = data_file
     
        
    def __getitem__(self, index):
        pass
       

    def __len__(self):
        pass

if __name__ == '__main__':
    dataset = LandmarkDatasetsCombined(data_file="/home/vuthede/AI/PFLD-pytorch/data/pfld_train_data_extra.rec")

    train_data_file = "/home/vuthede/AI/PFLD-pytorch/data/pfld_train_data_extra.rec"
    image_size=112
    batch_size=1
    train_iter = mx.io.ImageRecordIter(
        path_imgrec=train_data_file, 
        data_shape=(3, image_size, image_size), 
        batch_size=batch_size,
        label_width=205,
        shuffle = True,
        shuffle_chunk_size = 1024,
        seed = 1234, 
        prefetch_buffer = 10, 
        preprocess_threads = 16
    )

    print(len(train_iter))
    # for batch in train_iter:
    #     data   = batch.data[0].asnumpy()
    #     labels = batch.label[0].asnumpy()
    #     lmks = labels[:, 0:98*2]
    #     cate = labels[:, 2*98+1:2*98+6]
    #     angs = labels[:, -4:-1] 
        
    #     print(labels)
    #     break