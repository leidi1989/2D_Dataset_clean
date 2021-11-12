'''
Author: leidi
Date: 2020-10-19 13:30:55
LastEditTime: 2021-04-21 10:33:34
LastEditors: Leidi
Description: 将剪切的bbox图片拼接至目标数据集
FilePath: \Dataset_adjust_analyze_srcd:\Program\Dataset_deep_integrated\concate.py
'''
# -*- coding: utf-8 -*-
import argparse
import os
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import time

from Dataset_cleaning.utils.utils import *
from Dataset_cleaning.utils.extract_function import *
from Dataset_cleaning.utils.extract_bbox_function import *
from Dataset_cleaning.utils.concate_function import *


def concate_image_label(input_path_cut, input_path_tar, output_path, pick_classes_path, src_set_style, out_set_style, cut_image_list=None):

    cut_images_input_path = os.path.join(
        input_path_cut, 'cut_bbox_images')     # 获取bbox图片路径
    src_image_input_path = input_path_tar
    pickclasses_list = get_class(pick_classes_path)      # 读取挑选类别列表
    input_image_style = check_image_type(src_set_style)    # 获取输入图片类别
    input_label_style = check_src_lab_type(src_set_style)  # 获取输入label类别
    output_image_style = check_image_type(out_set_style)   # 获取输出图片类别
    output_label_style = check_src_lab_type(out_set_style)   # 获取输出label类别
    src_fold_list = pickup_fold_function(src_set_style)     # 获取源数据集组织结构列表
    target_class_list = get_class(get_names_list_path(
        os.path.join(input_path_tar, 'ImageSets')))   # 获取目标数据集类别列表

    cut_bbox_image_list = []
    if cut_image_list is None:      # 获取剪切true box图片的路径
        cut_bbox_image_list = os.listdir(
            cut_images_input_path)     # 获取bbox图片路径
        for i, one_cut_image in enumerate(cut_bbox_image_list):
            cut_bbox_image_list[i] = cut_images_input_path + \
                os.sep + one_cut_image
    else:
        cut_bbox_image_list = cut_image_list    # 获取剪切true box图片的路径

    total_concate_image_list = load_cut_image(
        cut_bbox_image_list)      # 读取剪切true box图片信息

    total_target_dateset_list = load_target_label(
        src_set_style, input_path_tar, target_class_list)    # 读取融合的目标数据集全部annotations信息
        
    # 展示图片true_box掩码图    
    for one in total_target_dateset_list:
        one.get_true_box_mask()
        cv2.imshow("self.mask", one.mask)
        cv2.waitKey(0)
        print(1)
        # one.get_free_space_rectangle()
        # cv2.imshow("self.mask", one.mask)
        # cv2.waitKey(0)
    
    allot_target_image(total_concate_image_list, total_target_dateset_list)

    print('1')
    
    return total_target_dateset_list


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='one_step.py')
    parser.add_argument('--cut', default=r'/mnt/hgfs/D/dataset/sjt_new_test_input_analyze',
                        type=str, help='cut bbox image dataset path')
    parser.add_argument('--tar', default=r'/mnt/hgfs/D/dataset/sjt_new_test_input_analyze_test',
                        type=str, help='target dataset path')
    parser.add_argument('--out', default=r'/mnt/hgfs/D/dataset/sjt_new_test_input_analyze_integrated',
                        type=str, help='output path')
    parser.add_argument('--istyle', '--is', dest='istyle', default=r'ldp',
                        type=str, help='source detaset style:ldp, pascal')
    parser.add_argument('--ostyle', '--os', dest='ostyle', default=r'ldp',
                        type=str, help='output detaset style:ldp, hy, yopascal,lov5')
    parser.add_argument('--pickclass', '--pc', dest='pickclass',
                        default=r'/mnt/hgfs/D/dataset/sjt_new_test_input_analyze/ImageSets/pick_classes.txt',
                        type=str, help='dataset class path')
    opt = parser.parse_args()

    input_path_cut = check_input_path(opt.cut)
    input_path_tar = check_input_path(opt.tar)
    output_path = check_output_path(opt.out)
    src_set_style = opt.istyle
    out_set_style = opt.ostyle
    pick_classes_path = check_input_path(opt.pickclass)     # 声明挑选类别文件路径

    print('Start concate images and lanels:\n')
    concate_image_label(input_path_cut, input_path_tar, output_path,
                        pick_classes_path, src_set_style, out_set_style)
    
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
