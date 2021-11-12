'''
Description: 
Version: 
Author: Leidi
Date: 2021-07-09 10:19:01
LastEditors: Leidi
LastEditTime: 2021-08-05 16:57:33
'''
# -*- coding:utf-8 -*-
import argparse
import os

from utils.utils import *
from utils.distribution_function import *


def distribution(output_path):
    """分析数据集的分布，包含total、train、val、test"""

    ImageSets_input_path = check_output_path(os.path.join(
        output_path, 'ImageSets'))     # 获取数据集ImageSets路径
    class_path = check_output_path(ImageSets_input_path, 'classes.names')     # 获取数据集类别文件路径    
    
    class_list = get_class(class_path)
    
    ttvt_path_list, label_input_path = get_path(output_path)  # 获取不同set的txt文件路径列表
    every_set_label_list = get_one_set_label_path_list(ttvt_path_list)      # 获取每个set.txt文件下图片的标签地址列表  
    set_count_dict_list, set_prop_dict_list = make_each_class_count_dict(label_input_path, every_set_label_list, class_list, ImageSets_input_path)    #生成不同set的计数字典   
    drow(set_count_dict_list, set_prop_dict_list, class_list, ImageSets_input_path)     # 在同图片中绘制不同set类别分布柱状图


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cleaning_7_distribution.py')
    parser.add_argument('--out', default=r'/home/leidi/Dataset/hy_highway_myxb_sjt_coco2017_7_classes_output_20210805',
                        type=str, help='output path')
    opt = parser.parse_args()

    output_path = check_output_path(opt.out)

    print('\nStart to distribution dataset：')
    distribution(output_path)
    print('Dataset analyze done!')

