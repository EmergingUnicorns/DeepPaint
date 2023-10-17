import cv2
import numpy as np
from PIL import Image

def read_img_rgb(path, resize = None):
    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    if (resize is not None):
        img = cv2.resize(img, resize)
    return img

def convert_numpy_to_PIL(l):
    p = []
    for i in l:
        p.append(Image.fromarray(i))
    return p

def convert_PIL_to_numpy(l):
    p = []
    for i in l:
        p.append(np.array(i))
    return p