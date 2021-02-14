import mxnet as mx
import cv2
import numpy as np

data_file_train_LP_rec = "/home/vuthede/data/pfld_1.0_68_style_LP_VW/pfld_train_data_LP.rec"
data_file_train_Style_rec = "/home/vuthede/data/pfld_1.0_68_style_LP_VW/pfld_train_data_style.rec"
data_file_train_VW_rec = "/home/vuthede/dat1/pfld_1.0_68_style_LP_VW/pfld_train_data_VW.rec"
data_file_valid_LP_rec = "/home/vuthede/dat1/pfld_1.0_68_style_LP_VW/pfld_test_data_LP.rec"
data_file_valid_Style_rec = "/home/vuthede/dat1/pfld_1.0_68_style_LP_VW/pfld_test_data_style.rec"
data_file_valid_VW_rec = "/home/vuthede/dat1/pfld_1.0_68_style_LP_VW/pfld_test_data_VW.rec"

image_size=192
batch_size=1

train_iter_LP = mx.io.ImageRecordIter(
        path_imgrec=data_file_train_LP_rec, 
        data_shape=(3, image_size, image_size), 
        batch_size=batch_size,
        label_width=139,
        shuffle = True,
        shuffle_chunk_size = 1024,
        seed = 1234, 
        prefetch_buffer = 10, 
        preprocess_threads = 2
    )

def plot_keypoints(img, lmks, color=(255,0,0)):
    for lm in lmks:
        lm = np.array(lm)*192
        lm = lm.astype(int)
        img = img.astype(np.uint8)
        print(lm)
        print(img.shape)
        cv2.circle(img, tuple(lm), 1, color, 1)
    return img


for batch in train_iter_LP:    
    img   = batch.data[0].asnumpy()
    labels = batch.label[0].asnumpy()
    landmark_gt = labels[:, 0:68*2]

    img = np.transpose(img[0], (1,2,0))
    img = np.ascontiguousarray(img)
    lmks = np.reshape(landmark_gt, (-1, 2))
    print(f'img shape: {img.shape}. Lmks shape: {lmks.shape}')
    img = plot_keypoints(img, lmks)

    cv2.imshow("Img", img)

    if cv2.waitKey(0)==27:
        break

cv2.destroyAllWindows()