'''
Description:
Version:
Author: Leidi
Date: 2021-10-27 14:05:30
LastEditors: Leidi
LastEditTime: 2022-02-18 14:58:16
'''
import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import namedtuple


image_path = r'/mnt/data_1/Dataset/dataset_temp/TI_edgeailite_auto_annotation_20220218/source_dataset_annotations/zhuoer20211124_000000_000006_leftImg8bit.png'

image = cv2.imread(image_path)


print(np.max(image))
print(np.min(image))
# image = cv2.resize(image, (1280, 720))
cv2.imshow('one', image)
cv2.waitKey(0)
