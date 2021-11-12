'''
Description: 
Version: 
Author: Leidi
Date: 2021-05-19 09:20:40
LastEditors: Leidi
LastEditTime: 2021-07-28 11:22:27
'''
# 批量修改图片文件名
# -*- coding: utf-8 -*-
import argparse
import os
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append('/home/leidi/Workspace/2D_Dataset_clean/Dataset_cleaning')
from utils.utils import *


def png_to_jpg(output_path):
    """[将image文件夹内png格式图片转换为jpg格式]

    Args:
        output_path ([str]): [数据集路径]
    """    
    print('\nStart change png to jpg:')
    output_path_images = check_output_path(output_path, 'JPEGImages')
    images_count = 0
    for one_image in tqdm(os.listdir(output_path_images)):
        if one_image.split('.')[-1] != 'png':
            continue
        in_image = os.path.join(output_path_images, one_image)
        out_image = os.path.join(output_path_images, one_image.replace('.png', '.jpg'))
        img = Image.open(in_image)
        img_split = img.split()
        if len(img.split()) == 4:
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(out_image, quality=100)
            images_count += 1
            os.remove(in_image)
        else:
            img.convert('RGB').save(out_image, quality=100)
            images_count += 1
            os.remove(in_image)
    print("Total pictures: %d images have been change." % images_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='tool_png_to_jpg.py')
    parser.add_argument('--out', default=r'D:\dataset\hy_dataset_202210121\hy_highway_truck_input',
                        type=str, help='output path')
    opt = parser.parse_args()

    output_path = check_output_path(opt.out)

    print('\nStart png_to_jpg:')
    png_to_jpg(output_path)
    print('\nPng_to_jpg done!\n')
