'''
Description: 
Version: 
Author: Leidi
Date: 2021-05-19 09:20:40
LastEditors: Leidi
LastEditTime: 2021-07-28 11:22:10
'''
'''
Description: 
Version: 
Author: Leidi
Date: 2021-05-16 02:39:30
LastEditors: Leidi
LastEditTime: 2021-06-17 17:28:17
'''
import os
import cv2
import argparse
from tqdm import tqdm

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
    opt = parser.parse_args()

    input_path = check_input_path(opt.set)
    output_root_path = check_output_path(opt.out)

    count = 0
    for root, dirs, files in tqdm(os.walk(input_path)):
        for image in tqdm(files):
            img_path = os.path.join(root, image)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            image_output_path = os.path.join(
                output_root_path, opt.pref + image)
            cv2.imwrite(image_output_path, img)
            count += 1
    print('\n合计转换图片数量：', count)
