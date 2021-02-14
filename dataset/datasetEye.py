import sys
sys.path.insert(0, "../GazeML/src")
from datasources import UnityEyes
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2
from torch.utils import data
import torch

def draw_gaze(image_in, eye_pos, pitchyaw, length=15.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

def draw_landmarks(img, lmks):
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, (255,0,0), -1, lineType=cv2.LINE_AA)

    return img



class UnityEyeDataset(data.Dataset):
    def __init__(self, data_dir, augmentation=True):
        gpu_options = tf.GPUOptions(allow_growth=True)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Declare some parameters
        self.elg_first_layer_stride = 1

        self.EYE_IMAGE_SHAPE = (128, 128)

        self.unityeyes = UnityEyes(
                    session,
                    batch_size=10,
                    data_format='NCHW',
                    unityeyes_path=data_dir,
                    min_after_dequeue=1000,
                    generate_heatmaps=True,
                    shuffle=True,
                    staging=False,
                    eye_image_shape= self.EYE_IMAGE_SHAPE,
                    heatmaps_scale=1.0 / self.elg_first_layer_stride,
                )
        
        if augmentation:
            self.unityeyes.set_augmentation_range('translation', 2.0, 10.0)
            self.unityeyes.set_augmentation_range('rotation', 1.0, 10.0)
            self.unityeyes.set_augmentation_range('intensity', 0.5, 20.0)
            self.unityeyes.set_augmentation_range('blur', 0.1, 1.0)
            self.unityeyes.set_augmentation_range('scale', 0.01, 0.1)
            self.unityeyes.set_augmentation_range('rescale', 1.0, 0.5)
            self.unityeyes.set_augmentation_range('num_line', 0.0, 2.0)
            self.unityeyes.set_augmentation_range('heatmap_sigma', 7.5, 2.5)

    def __len__(self):
        return self.unityeyes.num_entries

    
    def __getitem__(self, index):
        entry = self.unityeyes.entry_gen_by_index(index)
        data = self.unityeyes.preprocess_entry(entry)
        eye = data['eye']
        lmks = data['landmarks']

        #Lmks 2, 6 --> vertical eye
        #lmks 0, 4 --> horizaoontal eye
        gaze = data['gaze']
        
        v1 = np.linalg.norm(lmks[2]-lmks[6])
        v2 = np.linalg.norm(lmks[1]-lmks[7])
        v3 = np.linalg.norm(lmks[3]-lmks[5])

        h = np.linalg.norm(lmks[4]-lmks[0])
        ear_ratio = np.max([v1,v2,v3])/(h)

        norm_lmks = lmks/np.array(self.EYE_IMAGE_SHAPE)

        return torch.FloatTensor(eye), torch.FloatTensor([gaze[0], gaze[1], ear_ratio]), torch.FloatTensor(norm_lmks[:8].flatten())



# dataseteye = UnityEyeDataset(data_dir='/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/UNITYEYE/UnityEyes/imgs')

# print(len(dataseteye))

# import numpy as np

# while True:
#     index = np.random.randint(0, len(dataseteye)-1)
#     print(index)
#     eye, (g1, g2, ear), lmks = dataseteye[index]
#     eye = eye.numpy()
#     g1 = g1.numpy()
#     g2 = g2.numpy()
#     ear = ear.numpy()
#     lmks = lmks.numpy()
#     lmks = np.reshape(lmks, (8,2))
#     # print(f"Min eye: {np.min(eye)}. Max eye :{np.max(eye)}..")
#     eye += 1.0
#     eye *= 255.0
#     eye /= 2.0
#     eye = (np.transpose(eye, (1,2,0))).astype(np.uint8)
#     # print(f"Min eye: {np.min(eye)}. Max eye :{np.max(eye)}.")

#     print("Eye openess: ", ear)
#     eye = draw_gaze(eye, (32, 32), [g1, g2])
#     eye = draw_landmarks(eye, lmks*128.0)
#     # print(gaze*180)

#     cv2.imshow("Eye", eye)

#     k = cv2.waitKey(0)

#     if k==27:
#         break

#     cv2.destroyAllWindows()





# def ear(lmks):
#     v = np.linalg.norm(lmks[2]-lmks[6])
#     h = np.linalg.norm(lmks[4]-lmks[0])

#     return v/h


# print("Len dataset: ", unityeyes.num_entries)

# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# # Declare some parameters
# elg_first_layer_stride = 1

# unityeyes = UnityEyes(
#             session,
#             batch_size=10,
#             data_format='NCHW',
#             unityeyes_path="/media/vuthede/7d50b736-6f2d-4348-8cb5-4c1794904e86/home/vuthede/data/UNITYEYE/UnityEyes/imgs",
#             min_after_dequeue=1000,
#             generate_heatmaps=True,
#             shuffle=True,
#             staging=False,
#             eye_image_shape=(224, 224),
#             heatmaps_scale=1.0 / elg_first_layer_stride,
#         )

# for entry in unityeyes.entry_generator():
#     data = unityeyes.preprocess_entry(entry)
#     eye = data['eye']
#     lmks = data['landmarks']

#     #Lmks 2, 6 --> vertical eye
#     #lmks 0, 4 --> horizaoontal eye
#     gaze = data['gaze']
#     ear_ratio = ear(lmks)
#     # print(f"Eye shape: {eye.shape}. Lmks shape :{lmks.shape}. Gaze shape: {gaze.shape}")
#     print(f"EAR: ", ear_ratio)

#     eye = np.transpose(eye, (1,2,0))
#     eye = draw_gaze(eye, (30, 18), gaze)
#     eye = draw_landmarks(eye, lmks[:8])
#     print(gaze*180)

#     cv2.imshow("Eye", eye)

#     k = cv2.waitKey(0)

#     if k==27:
#         break

# cv2.destroyAllWindows()


    