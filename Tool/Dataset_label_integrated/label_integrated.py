'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-21 10:02:07
LastEditors: Leidi
LastEditTime: 2021-04-21 10:43:09
'''
# -*- coding: utf-8 -*-
import argparse
import os
from tqdm import tqdm
import time

from Dataset_label_integrated.utils.utils import *
from Dataset_label_integrated.utils.fix_function import *
from Dataset_label_integrated.utils.label_output import *
from Dataset_label_integrated.utils.extract_function import *
from Dataset_label_integrated.utils.output_path_function import *
from Dataset_label_integrated.utils.input_path_function import *
from Dataset_label_integrated.utils.ImageSets import *

import Dataset_label_integrated.label_integrated_1_copy_all


def label_integrated(input_path, output_path, src_class_path, fix_class_file_path, src_set_style, out_set_style):
    
    
    image_type = check_image_type(src_set_style)    # 获取源数据集类别对应图片类别
    classes_src_list = get_class(src_class_path)    # 获取源数据集类别
    classes_fix_dict = get_fix_class_dict(fix_class_file_path)    # 获取修改后类别字典
    new_class_names_list = get_new_class_names_list(classes_fix_dict)   # 获取融合后的类别列表
    output_path_ann = check_output_path(os.path.join(output_path, "Annotations"))
    # output_label_style = out_set_style
    print('\nStart to extract dataset annotations:')
    data_list = pickup_data_from_function(src_set_style, input_path, classes_src_list)

    print('\nStart to fix class in annotations:')
    # 获取全部标签名称列表，修改标签后key为标签名称value为对应路径的全部标签字典
    data_list = fix_annotation(data_list, classes_fix_dict)

    print('\nStart to creat mix annotations:')
    pickup_data_out_function(out_set_style, output_path_ann, data_list)
    
    return data_list


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='label_integrated.py')
    parser.add_argument('--set', default=r'E:\2.Datasets\test_output',
                        type=str, help='dataset path')
    parser.add_argument('--names', '--sn', dest='names',
                        default=r'E:\2.Datasets\test_output\ImageSets\classes.names',
                        type=str, help='dataset class path')
    parser.add_argument('--fixnamesfile', '--fn', dest='fixnamesfile',
                        default=r'D:\Program\2D_Dataset_fix\Dataset_label_integrated\data\fix_classes.txt',
                        type=str, help='dataset class path')
    parser.add_argument('--out', default=r'E:\2.Datasets\test_output_label_integrated',
                        type=str, help='output path')
    parser.add_argument('--istyle', '--is', dest='istyle', default=r'ldp',
                        type=str, help='source detaset style: ldp')
    parser.add_argument('--ostyle', '--os', dest='ostyle', default=r'ldp',
                        type=str, help='output detaset style: ldp')
    opt = parser.parse_args()

    input_path = check_input_path(opt.set)
    output_path = check_output_path(opt.out)
    src_class_path = opt.names
    fix_class_file_path = opt.fixnamesfile
    src_set_style = opt.istyle
    out_set_style = opt.ostyle

    label_integrated(input_path, output_path, src_class_path, fix_class_file_path, src_set_style, out_set_style)
    
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
