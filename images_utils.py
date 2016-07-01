__author__ = 'zarbo'
import cv2
import numpy as np
def load_im2(paths):
    l = []
    for name in paths:
        im2 = cv2.resize(cv2.imread(name), (224, 224)).astype(np.float32)
        im2[:, :, 0] -= 103.939
        im2[:, :, 1] -= 116.779
        im2[:, :, 2] -= 123.68
        im2 = im2.transpose((2, 0, 1))
        # im2 = np.expand_dims(im2, axis=0)
        # print(im2.shape)
        l.append(im2)
    return l