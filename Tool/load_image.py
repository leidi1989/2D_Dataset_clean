'''
Description:
Version:
Author: Leidi
Date: 2021-10-27 14:05:30
LastEditors: Leidi
LastEditTime: 2022-02-10 16:35:34
'''
import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import namedtuple


image_path = r'/mnt/data_1/Dataset/dataset_temp/clean/yunce_wudazhuoer_parking_20211213_20220209/cityscapes/data/gtFine/train/yunce/yunce_000000_000932_gtFine_labelIds.png'

image = cv2.imread(image_path)


print(np.max(image))
print(np.min(image))
# image = cv2.resize(image, (1280, 720))
cv2.imshow('one', image)
cv2.waitKey(0)
