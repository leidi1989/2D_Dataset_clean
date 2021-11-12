'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-21 10:02:07
LastEditors: Leidi
LastEditTime: 2021-04-21 10:41:38
'''
# -*- coding: utf-8 -*-
import os
import argparse
from tqdm import tqdm
import shutil
import time

from Dataset_label_integrated.utils.utils import *
from Dataset_label_integrated.utils.output_path_function import *
from Dataset_label_integrated.utils.input_path_function import *


def copy_images_annotations(src_put_path_dict, output_path_dict, src_set_style, out_set_style, total_mix_images_name_list):
    """向新数据集路径拷贝image、annotation、"""

    image_type = check_image_type(src_set_style)
    # 复制源数据集images下图片、annotation下xml文件至融合数据集下
    for one_name in tqdm(total_mix_images_name_list):
        # 配置image、annotation名字
        src_image_path = os.path.join(
            src_put_path_dict['images_path'], one_name + '.' + image_type)
        src_Annotation_path = os.path.join(
            src_put_path_dict['annotations_path'], one_name + '.xml')
        out_image_path = os.path.join(
            output_path_dict['images_path'], one_name + '.' + image_type)
        out_Annotation_path = os.path.join(
            output_path_dict['annotations_path'], one_name + '.xml')
        # 复制image、annotation
        shutil.copyfile(src_image_path, out_image_path)
        shutil.copyfile(src_Annotation_path, out_Annotation_path)

    for n in tqdm(os.listdir(src_put_path_dict['source_label_path'])):
        src_source_label_path = os.path.join(
            src_put_path_dict['source_label_path'], n)
        out_source_label_path = os.path.join(
            output_path_dict['source_label_path'], n)
        shutil.copyfile(src_source_label_path, out_source_label_path)


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='copy_all.py')
    parser.add_argument('--set', default=r'D:\DataSets\hy_mix_dataset_yolo_input_analyze',
                        type=str, help='dataset path')
    parser.add_argument('--out', default=r'D:\DataSets\test1_analyze',
                        type=str, help='output path')
    parser.add_argument('--istyle', '--is', dest='istyle', default=r'hy',
                        type=str, help='source detaset style: hy')
    parser.add_argument('--ostyle', '--os', dest='ostyle', default=r'hy',
                        type=str, help='output detaset style: hy, yolov5')
    opt = parser.parse_args()

    input_path = check_input_path(opt.set)
    output_path = check_output_path(opt.out)
    src_set_style = opt.istyle
    out_set_style = opt.ostyle

    src_set_fold_list = pickup_fold_function(
        src_set_style)     # 获取对应数据集类型的文件路径
    for one in src_set_fold_list:       # 创建融合数据集目录
        check_output_path(os.path.join(output_path, one))

    src_put_path_dict = pickup_src_set_output_function(
        src_set_style, input_path)     # 获取源数据集下文件组织路径
    output_path_dict = pickup_new_set_output_function(
        out_set_style, output_path)   # 获取新数据集下文件组织路径

    total_mix_images_name_list = []     # 声明混合数据集图片路径列表

    with open(os.path.join(output_path_dict['imagesets_path'], 'total_image_name.txt')) as total_mix_images_name:
        for one_line in total_mix_images_name.writelines():
            one_line = one_line.replace('\n', '')
            total_mix_images_name_list.append(one_line)

    print('\nStart to copy images and annotations:')
    copy_images_annotations(src_put_path_dict, output_path_dict,
                            src_set_style, out_set_style, total_mix_images_name_list)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')
