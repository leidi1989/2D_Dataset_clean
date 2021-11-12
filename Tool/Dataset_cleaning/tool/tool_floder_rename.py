'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:03
LastEditors: Leidi
LastEditTime: 2021-05-15 17:56:12
'''
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
from tqdm import tqdm

import sys
sys.path.append('/home/leidi/Workspace/2D_Dataset_clean/Dataset_cleaning')
from utils.utils import *
from utils.image_function import *


def floder_rename(input_path, output_path, name_pre, input_style, layer):
    """[移动原数据集内图片，使用文件夹名称更名文件，存入输出路径]

    Parameters
    ----------
    input_path : [str]
        [源数据集输入路径]
    output_path : [str]
        [输出数据集路径]
    name_pre : [str]
        [名称前缀]
    input_style : [str]
        [输入数据集类型]
    layer : [int]
        [按文件夹路径更名层级]
    """

    output_path_images = check_output_path(output_path, 'JPEGImages')
    output_path_labels = check_output_path(output_path, 'labels')

    images_count, labels_count = 0, 0
    if not name_pre:
        name_pre=''     # 判断是否需要添加文件前缀
    for root, dirs, files in os.walk(input_path):
        for fileName in tqdm(files):   # 遍历文件夹中所有文件
            re_fileName = pickup_image_from_function(
                input_style, root, fileName)

            # 判断图片格式
            if fileName[-len(check_image_type(input_style)):] == check_image_type(input_style):
                source_file = os.path.join(root, fileName)     # 源文件
                floder = os.path.split(source_file)[0]  # 读取文件所在文件夹路径
                floder = floder.split(os.sep)   # 按分隔符切分文件所在文件夹路径
                floder_str = ''
                if layer != 0:
                    for i, one_floder in enumerate(floder[layer:]):     # 拼接切分的文件夹路径
                        floder_str += one_floder + '_'
                        # if i+1 != len(floder[layer:]):
                        #     floder_str += '_'
                rename_file = os.path.join(
                    output_path_images, name_pre + floder_str + fileName)    # 以文件所在文件夹路径，按层级改名后名称
                if os.path.exists(rename_file):
                    os.remove(rename_file)
                shutil.copyfile(source_file, rename_file)
                images_count += 1

            # 判断标签格式
            elif fileName[-len(check_src_lab_type(input_style)):] == check_src_lab_type(input_style):
                source_file = os.path.join(root, fileName)     # 源文件
                floder = os.path.split(source_file)[0]
                floder = floder.split(os.sep)
                floder_str = ''
                if layer != 0:
                    for i, one_floder in enumerate(floder[layer:]):     # 拼接切分的文件夹路径
                        floder_str += one_floder + '_'
                        # if i+1 != len(floder[layer:]):
                        #     floder_str += '_'
                rename_file = os.path.join(
                    output_path_labels, name_pre + floder_str + fileName)    # 以文件所在文件夹路径，按层级改名后名称
                if os.path.exists(rename_file):
                    os.remove(rename_file)
                shutil.copyfile(source_file, rename_file)
                labels_count += 1

    print("Total pictures: %d images, labels: %d labels, have been saved to %s and %s." % (
        images_count, labels_count, output_path_images, output_path_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='images.py')
    parser.add_argument('--set', default=r'D:\DataSets\20000张街景图片目标检测数据\data',
                        type=str, help='dataset path')
    parser.add_argument('--out', default=r'D:\DataSets\sjt_dataset_input',
                        type=str, help='output path')
    parser.add_argument('--pref', default='',
                        type=str, help='rename prefix')
    parser.add_argument('--layer', default='0',
                        type=str, help='rename prefix')
    parser.add_argument('--ilstyle', '--is', dest='ilstyle', default=r'sjt',
                        type=str, help='input labels style: ldp, hy, voc, kitti, cctsdb, lisa, hanhe, sjt')
    opt = parser.parse_args()

    input_path = check_input_path(opt.set)
    output_path = check_output_path(opt.out)
    name_pre = check_pref(opt.pref)
    layer = int(opt.layer)
    input_style = opt.ilstyle
    
    

    print('\nStart images change:')
    floder_rename(input_path, output_path, name_pre, input_style, layer)
    print('\nChange each image file done!\n')
