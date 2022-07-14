'''
Description:
Version:
Author: Leidi
Date: 2021-10-27 14:05:30
LastEditors: Leidi
LastEditTime: 2022-06-26 10:42:56
'''
import cv2
import numpy as np

image_path = r'/home/leidi/Desktop/bev_1.png'

image = cv2.imread(image_path)

print(np.max(image))
print(np.min(image))
image = cv2.resize(image, (1280, 1280))
cv2.imshow('one', image)
cv2.waitKey(0)
