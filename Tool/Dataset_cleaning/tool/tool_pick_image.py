'''
Description: 
Version: 
Author: Leidi
Date: 2021-06-17 16:50:56
LastEditors: Leidi
LastEditTime: 2021-06-17 17:28:30
'''
import os
import shutil
import argparse
from tqdm import tqdm
import numpy as np

import sys
sys.path.append('/home/leidi/Workspace/2D_Dataset_clean/Dataset_cleaning')
from utils.utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cleaning_5_labels.py')
    parser.add_argument('--set', default=r'/mnt/disk1/detect_sample/telescope/20210616/img_20210616_2',
                        type=str, help='dataset path')
    parser.add_argument('--out', default=r'/mnt/disk1/detect_sample/telescope/20210616/img_20210616_3',
                        type=str, help='output path')
    parser.add_argument('--pref', default=r'',
                        type=str, help='rename prefix')
    parser.add_argument('--fps', default=25,
                        type=str, help='rename prefix')
    opt = parser.parse_args()

    img_path = check_output_path(opt.set)
    profix = opt.pref
    output_path = check_output_path(opt.out)
    fps = opt.fps
    
    image_list = os.listdir(img_path)
    image_index = np.arange(0, len(image_list))
    image_index_list = []
    
    print('Count pickup image index:')
    for n, velue in tqdm(enumerate(image_index)):
        if 0 == n % 25 or 0 == n:
            image_index_list.append(n)

    print('Pickup image :')
    for one in tqdm(image_index_list):
        image_src_path = os.path.join(img_path, image_list[one])
        image_dir_path = os.path.join(output_path, profix + image_list[one])
        shutil.copy(image_src_path, image_dir_path)

    print('End.')
