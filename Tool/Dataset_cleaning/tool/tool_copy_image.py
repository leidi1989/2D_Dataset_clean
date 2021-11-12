'''
Description: 
Version: 
Author: Leidi
Date: 2021-06-29 13:52:31
LastEditors: Leidi
LastEditTime: 2021-07-28 11:21:33
'''
import os
from tqdm import tqdm
import shutil

import sys
sys.path.append('/home/leidi/Workspace/2D_Dataset_clean/Dataset_cleaning')
from utils.utils import *


input_path = check_input_path(r'/home/leidi/Desktop/images')
output_path = check_output_path(r'/home/leidi/Desktop/images_rename')

image_list = os.listdir(input_path)
for n, one in enumerate(tqdm(image_list)):
    input_image_path = os.path.join(input_path, one)
    output_image_path = os.path.join(output_path, '20210629_' + str(n) + '.jpg')
    shutil.copy(input_image_path, output_image_path)
