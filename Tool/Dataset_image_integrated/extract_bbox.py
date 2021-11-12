'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-21 10:02:06
LastEditors: Leidi
LastEditTime: 2021-07-13 16:03:08
'''
# -*- coding: utf-8 -*-
import argparse
import os
import cv2
from tqdm import tqdm
import time

from utils.utils import *
from utils.extract_function import *
from utils.extract_bbox_function import *


def extract_bbox(input_path, src_set_style, pick_classes_path, out_set_style, image_resize, save_image, image_expand):
    """[提取源输入图片bbox，并创建bbox图片，命名为原图片名称_bboxclassname_number.jpg]

    Parameters
    ----------
    input_path : [str]
        [剪切true box图片数据集]
    src_set_style : [str]
        [剪切true box图片数据集类型]
    pick_classes_path : [str]
        [挑选剪切true box类别]
    out_set_style : [str]
        [输出类别]
    save_image : [str]
        [是否保存提取图片]

    Returns
    -------
    cut_image_list : [list]
        [提取true box图片数据]
    """

    output_image_path = check_output_path(
        input_path, 'cut_bbox_images')    # 剪切图片输出路径
    putout_image_style = check_image_type(out_set_style)    # 输出剪切图片格式
    pickclasses_list = get_class(pick_classes_path)      # 读取挑选类别列表
    class_list = get_class(get_names_list_path(input_path))     # 获取类别列表

    data_list = []      # 声明数据集信息列表
    data_list = pickup_data_from_function(
        src_set_style, input_path, class_list)    # 抽取源标签图像数据信息
    # 依据抽取的data_list信息截取源图片bbox，并将bbox从原图中裁切出来
    n = creat_bbox_image(
        data_list, putout_image_style, output_image_path, pickclasses_list, image_expand, image_resize, save_image)
    print('Cut bbox end, total images:{0}'.format(n))

    return data_list


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='extract_bbox.py')
    parser.add_argument('--set', default=r'/home/leidi/Dataset/ccpd_base_detect_input_analyze',
                        type=str, help='dataset path')
    parser.add_argument('--istyle', '--is', dest='istyle', default=r'ldp',
                        type=str, help='source detaset style:ldp, pascal')
    parser.add_argument('--ostyle', '--os', dest='ostyle', default=r'ldp',
                        type=str, help='output detaset style: ldp, hy, yolov5')
    parser.add_argument('--pickclasses', '--pc', dest='pickclasses',
                        default=r'/home/leidi/Dataset/ccpd_base_detect_input_analyze/classes.names',
                        type=str, help='dataset class path. (.txt、.names)')
    parser.add_argument('--expand', '--ep', dest='expand', default=0.00,
                        type=float, help='resize cut images, 0-1.')
    parser.add_argument('--resize', '--rs', dest='resize', default=1.0,
                        type=float, help='resize cut images, 0-inf.')
    parser.add_argument('--saveimg', '--si', dest='saveimg', default=1,
                        type=str, help='save cut images.')
    opt = parser.parse_args()

    input_path = check_input_path(opt.set)
    pick_classes_path = check_input_path(opt.pickclasses)
    src_set_style = opt.istyle
    out_set_style = opt.ostyle
    save_image = opt.saveimg
    image_resize = opt.resize
    image_expand = opt.expand

    print('Start cut bbox, and output images:')
    cut_image_list = extract_bbox(input_path, src_set_style, pick_classes_path,
                                  out_set_style, image_resize, save_image, image_expand)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')
