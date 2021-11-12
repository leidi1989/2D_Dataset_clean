'''
Description: 
Version: 
Author: Leidi
Date: 2021-07-09 10:19:01
LastEditors: Leidi
LastEditTime: 2021-10-21 20:08:45
'''
# 批量修改图片文件名
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
from tqdm import tqdm

from utils.utils import *
from utils.image_function import *


def images_move(input_path, output_path, name_pre, input_label_style):
    """[移动原数据集内图片，并更名存入输出路径]

    Parameters
    ----------
    input_path : [str]
        [源数据集路径]
    output_path : [str]
        [输出数据集路径]
    name_pre : [str]
        [添加文件名称的前缀]
    input_label_style : [str]
        [输入数据集标签类别]
    """
    
    output_path_images = check_output_path(output_path, 'JPEGImages')
    classes_names_path = get_names_list_path(input_path)
    classes_names_output_path = os.path.join(output_path, 'classes.names')
    if os.path.exists(classes_names_output_path):
        os.remove(classes_names_output_path)
    shutil.copyfile(classes_names_path, classes_names_output_path)

    images_count = 0
    for root, dirs, files in os.walk(input_path):
        for fileName in tqdm(files):   # 遍历文件夹中所有文件
            re_fileName = pickup_image_from_function(
                input_label_style, root, fileName)  # 判断图片命名格式
            # 判断图片格式
            if fileName.split('.')[-1] == check_image_type(input_label_style):
                source_file = os.path.join(root, fileName)     # 源文件
                if name_pre == None:
                    name_pre = ''
                rename_file = os.path.join(
                    output_path_images, name_pre + str(re_fileName))    # 改名后名称
                if os.path.exists(rename_file):
                    os.remove(rename_file)
                rename_file = check_image_rename(input_label_style, rename_file)
                shutil.copyfile(source_file, rename_file)
                images_count += 1
    png_to_jpg(output_path)
    print("Total pictures: %d images have been saved to %s." %
          (images_count, output_path_images))


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='cleaning_1_images.py')
    parser.add_argument('--set', default=r'/home/leidi/Dataset/ccpd_test',
                        type=str, help='dataset path')
    parser.add_argument('--out', default=r'/home/leidi/Dataset/ccpd_test_analyze',
                        type=str, help='output path')
    parser.add_argument('--pref', default=r'ccpd',
                        type=str, help='rename prefix')
    parser.add_argument('--segment', '--sg', dest='segment', default='_',
                        type=str, help='name split')
    parser.add_argument('--ilstyle', '--is', dest='ilstyle', default=r'ccpd',
                        type=str, help='input labels style: ldp, hy, myxb, nuscenes, \
                                                            pascal, hy_highway, coco2017, \
                                                            kitti, cctsdb, lisa, \
                                                            hanhe，yolov5_detect, yolo, \
                                                            sjt, ccpd')
    opt = parser.parse_args()

    input_path = check_input_path(opt.set)
    output_path = check_output_path(opt.out)
    input_label_style = opt.ilstyle
    segment = opt.segment
    name_pre = check_pref(opt.pref, segment)

    print('\nStart images change:')
    images_move(input_path, output_path, name_pre, input_label_style)
    print('\nChange each image file done!\n')

    time_end = time.time()
    print('time cost', time_end-time_start, 's')
