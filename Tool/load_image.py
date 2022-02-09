'''
Description:
Version:
Author: Leidi
Date: 2021-10-27 14:05:30
LastEditors: Leidi
LastEditTime: 2021-10-31 14:29:09
'''
import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import namedtuple


image_path = r'/media/leidi/hy_dataset/Dataset/huawei_dataset_total/hy_segement_dataset_huawei_3215_20210918_20211031/cityscapes/data/leftImg8bit/val/huawei3215/huawei3215_000000_000145_leftImg8bit.png'

image = cv2.imread(image_path)


print(np.max(image))
print(np.min(image))
# image = cv2.resize(image, (1280, 720))
cv2.imshow('one', image)
cv2.waitKey(0)
