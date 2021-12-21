'''
Description: 
Version: 
Author: Leidi
Date: 2021-07-09 10:19:01
LastEditors: Leidi
LastEditTime: 2021-12-21 16:15:03
'''
# -*- coding: utf-8 -*-
import argparse
import os
from tqdm import tqdm

from utils.utils import *
from utils.key_modify import *


def source_label_change_key(input_path, output_path, name_pre, input_label_style):
    """更换源标签文件中图片文件名的key的值"""

    output_path_src_lab = check_output_path(output_path, 'source_label')
    change_count = 0
    for root, dirs, files in os.walk(input_path):
        for fileName in tqdm(files):   # 遍历文件夹中所有文件
            if os.path.splitext(str(fileName))[-1] == check_src_lab_type(input_label_style):   # 判断label格式
                count_temp = pickup_move_function(input_label_style, 
                                                  output_path_src_lab, root, fileName, name_pre)   # 对应不同类别进行更名
                change_count += count_temp
    print("Total name change: %d save to %s." % (change_count, output_path_src_lab))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cleaning_2_keys.py')
    parser.add_argument('--set', default=r'/home/leidi/Dataset/hy_highway_myxb_sjt_coco2017_7_classes_input_20210805',
                        type=str, help='dataset path')
    parser.add_argument('--out', default=r'/home/leidi/Dataset/hy_highway_myxb_sjt_coco2017_7_classes_output_20210805',
                        type=str, help='output path')
    parser.add_argument('--pref', default=r'',
                        type=str, help='rename prefix')
    parser.add_argument('--segment', '--sg', dest='segment', default='_',
                        type=str, help='name split')
    parser.add_argument('--ilstyle', '--is', dest='ilstyle', default=r'ldp',
                        type=str, help='input labels style: ldp, hy, myxb, nuscenes, \
                                                            pascal, hy_highway, coco2017, \
                                                            kitti, cctsdb, lisa, \
                                                            hanhe, yolov5_detect, yolo, \
                                                            sjt, ccpd')
    opt = parser.parse_args()

    input_path = check_input_path(opt.set)
    output_path = check_output_path(opt.out)
    input_label_style = opt.ilstyle
    segment = opt.segment
    name_pre = check_pref(opt.pref, segment)

    print('\nStart source label change:')
    source_label_change_key(input_path, output_path, name_pre, input_label_style)
    print('\nChange each label done!\n')
