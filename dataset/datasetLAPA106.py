import numpy as np
import cv2
import sys
sys.path.append('..')

from torch.utils import data
import glob
import os
import mxnet as mx

def make_pfld_record(output, list_dataset, dest_size=192):
    record = mx.recordio.MXRecordIO(output, 'w')
    idx =  0
    for dataset in list_dataset:
        for img, landmarks in dataset:
            landmarks = np.reshape(landmarks, (-1,2))
            landmarks = lmks106_to_lmks68(landmarks)
            landmarks = np.reshape(landmarks, (-1))

            idx += 1  
            print(idx)
            img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = np.asarray(img).shape
            if (h!=dest_size and w !=dest_size):
                img = cv2.resize(img, (dest_size, dest_size))
            lmks = list(landmarks)
            #### No angle
            angles = []
            for i in range(0, 3):
                angles.append(
                    float(0.0)
                )
            label  = lmks + angles

            header = mx.recordio.IRHeader(0, label, i, 0)
            packed_s = mx.recordio.pack_img(header, img)
            record.write(packed_s)


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
    # TARGET_IMAGE_SIZE = (112, 112)
    TARGET_IMAGE_SIZE = (192, 192)

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
        # self.landmark = self._get_106_landmarks(f.replace(self.img_dir, self.anno_dir).replace(".jpg", ".txt"))

        replacing_extension = ".txt"
        # if "AFW" in os.path.basename(f) or \
        #    "HELEN" in os.path.basename(f) or \
        #    "IBUG" in os.path.basename(f) or \
        #    "LFPW" in os.path.basename(f):        
        #     replacing_extension = ".jpg.txt"

        self.landmark = self._get_106_landmarks(f.replace(self.img_dir, self.anno_dir).replace(".jpg", replacing_extension))


        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        boxsize = int(np.max(wh)*1.6)
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
        landmark_original = np.reshape(landmark_original, (-1)).astype(np.float32)
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
        landmark = np.reshape(landmark, (-1)).astype(np.float32)

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


def lmks106_to_lmks98(l):
    boundary_and_nose = list(range(0, 55))
    below_nose = list(range(58, 63))
    boundary_left_eye = list(range(66, 74))
    boundary_right_eye = list(range(75, 83))        
    mouth = list(range(84, 104))
    center_left_eye = [104]
    center_right_eye = [105]

    indice = boundary_and_nose + below_nose + boundary_left_eye + boundary_right_eye + mouth + center_left_eye +  center_right_eye
    l = np.array(l)[indice]

    return l

def lmks98_to_lmks68(l):
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

    indice68lmk = np.array(list(LMK98_2_68_MAP.keys()))
    l  = np.array(l)[indice68lmk]

    return l

def lmks106_to_lmks68(l):
    l = lmks106_to_lmks98(l)
    l = lmks98_to_lmks68(l)
    return l

# def preprocess(image_int):
#     image_float = image_int.astype('float32')/255 
#     # augs = mx.image.CreateAugmenter(data_shape=(3, args.image_size, args.image_size), 
#     #                                 brightness=0.125, contrast=0.125, saturation=0.125)
#     # for aug in augs:
#     #     image_float = aug(image_float)
#     normalizer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     image = normalizer(image_float)
#     return image


if __name__ == "__main__":



    # lapa = LAPA106DataSet(img_dir="/home/vuthede/data/LaPa/train/images",
    #                         anno_dir="/home/vuthede/data/LaPa/train/landmarks")
    
    # lapa_valid = LAPA106DataSet(img_dir="/home/vuthede/data/LaPa/valid/images",
    #                         anno_dir="/home/vuthede/data/LaPa/valid/landmarks")
    
    # lapa_test = LAPA106DataSet(img_dir="/home/vuthede/data/LaPa/test/images",
    #                         anno_dir="/home/vuthede/data/LaPa/test/landmarks")
    
    # afw = LAPA106DataSet(img_dir="/home/vuthede/data/Training_data/AFW/picture",
    #                         anno_dir="/home/vuthede/data/Training_data/AFW/landmark")
    # helen = LAPA106DataSet(img_dir="/home/vuthede/data/Training_data/HELEN/picture",
    #                         anno_dir="/home/vuthede/data/Training_data/HELEN/landmark")
    # ibug = LAPA106DataSet(img_dir="/home/vuthede/data/Training_data/IBUG/picture",
    #                         anno_dir="/home/vuthede/data/Training_data/IBUG/landmark")
    # lfpw = LAPA106DataSet(img_dir="/home/vuthede/data/Training_data/LFPW/picture",
    #                         anno_dir="/home/vuthede/data/Training_data/LFPW/landmark")

    # print(len(lapa), len(afw), len(helen), len(ibug), len(lfpw))

    # # make_pfld_record(output="./LAPA_RECORD/train_lapa_noangle.rec", list_dataset=[lapa],dest_size=192)
    # # make_pfld_record(output="./LAPA_RECORD/test_lapa_noangle.rec", list_dataset=[lapa_valid, lapa_test],dest_size=192)



    data = mx.io.ImageRecordIter(
        # path_imgrec="LAPA_RECORD/train_lapa_noangle.rec", 
        path_imgrec="/home/vuthede/data/pfld_1.0_68_style_LP_VW/pfld_train_data_VW.rec",
        data_shape=(3, 192, 192), 
        batch_size=2,
        label_width=139,
        shuffle = True,
        shuffle_chunk_size = 1024,
        seed = 1234, 
        prefetch_buffer = 10, 
        preprocess_threads = 1
    )

    for batch in data:
        # batch_idx += 1
        batch_size = batch.data[0].shape[0]

        data   = batch.data[0]
        labels = batch.label[0]
        lmks = labels[:, 0:68*2]

        img = data[0]
        img = np.transpose(img.asnumpy(), (1,2,0)).astype(np.uint8)
        lmk = lmks[0]

        lmk = lmk.asnumpy()
        lmk  = np.reshape(lmk, (68,2))*192
        lmk = lmk.astype(np.uint8)
        img1 = np.ones(img.shape)*255
        # img1 = img

        print(lmk)
        for p in lmk[[37, 41,38,40,36,39]]:
            cv2.circle(img1, tuple(p), 1, (255, 0, 0), 1)
        
        k =  cv2.waitKey(0)

        cv2.imshow("asd", img1)
        if k==27:
            break

    cv2.destroyAllWindows()


    # import matplotlib.pyplot as plt
    
    # img = np.transpose(img.asnumpy(), (1,2,0)).astype(np.uint8)
    # img = np.array(img).astype(np.uint8)
    # img = np.clip(img, 0, 255)

    # # print(img)
    # lmk = lmk.asnumpy()

    # lmk  = np.reshape(lmk, (68,2))*192
    # lmk = lmk.astype(np.uint8)
    # print(img.shape, lmk.shape)
    # # img = np.ones(img.shape)*255
    # for p in lmk:
    # #     print(type(img))
    # #     print(type(img[0][0][0]))
    #     print(tuple(p))
    #     cv2.circle(img, tuple(p), 1, (255, 0, 0), 1)
    
    # plt.imshow(img)
    # plt.show()

    # make_pfld_record(output="./LAPA_RECORD/train.rec", list_dataset=[ibug],dest_size=192)
    # img, lmks = lapa[19]
    # print(img.shape)
    # print(type(img))
    # print(type(img[0][0][0]))

    # print(lmks.shape)
    # lmks = np.reshape(lmks, (-1,2)) *192

    # lmks68 = lmks106_to_lmks68(lmks)

    # import matplotlib.pyplot as plt

    # # for p in lmks[[1,10,11,17, 98]]:
    # for p in lmks68[list(range(42, 49))]:
    #     p = p.astype(int)
    #     cv2.circle(img, tuple(p), 1, (255, 0, 0), 1)
    
    # plt.imshow(img)
    # plt.show()
