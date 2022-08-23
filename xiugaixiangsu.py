from PIL import Image
import numpy as np
import cv2
import os

path = r'E:\10\1/'
save_path = r'E:\10\6'
for i in os.listdir(path):
    img = np.array(Image.open(path + i))
    img[img==64]=1
    img[img== 127] = 2
    img[img== 191] = 3
    img[img== 255] = 4
    img[img== 85] = 1
    img[img== 170] = 2


    cv2.imwrite(os.path.join(save_path, i), img)
