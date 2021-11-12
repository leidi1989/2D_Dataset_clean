'''
Description: 
Version: 
Author: Leidi
Date: 2020-09-04 13:21:36
LastEditors: Leidi
LastEditTime: 2021-10-21 20:08:55
'''
# -*- coding: utf-8 -*-
import argparse
import os

from utils.utils import *
from utils.extract_function import *
from utils.label_function import *


def creat_label(output_path, input_label_style, output_label_style, model, class_path, pre_data_list):
    """将xml文件转换为对应格式的txt文件"""

    source_labels_input_path = output_path      # 源标签路径
    output_path_1 = check_output_path(output_path, 'labels')    # 输出labels路径
    output_path_2 = check_output_path(
        output_path, 'ImageSets')     # 输出ImageSets路径
    class_list = get_class(class_path)  # 获取类别列表
    data_list = pre_data_list.copy()    # extract获取的数据集类别、坐标数据

    total_label_list = []
    if len(data_list) == 0:
        data_list = pickup_data_from_function(
            input_label_style, source_labels_input_path, class_list)    # 读取源标签内图片信息
        data_list = cheak_total_images_data_list(data_list)   # 在总图片信息列表中剔除无真实框图片信息
    
    total_label_list = pickup_data_out_to_label_function(
        model, output_path_1, data_list, class_list)    # 生成对应格式的新labels

    with open(os.path.join(output_path_2, 'total_image_name.txt'), 'w') as total_label_out:   # 创建图片对应txt格式的label文件
        total_label_out.write('\n'.join(str(x) for x in total_label_list))

    with open(os.path.join(output_path_2, 'classes.names'), 'w') as class_list_out:   # 创建txt格式的classes.names文件
        class_list_out.write('\n'.join(str(x) for x in class_list))

    print("Total create labels : %d save to %s." %
          (len(data_list), output_path_1))
    
    print('Start delete empty bbox images:')
    delete_empty_images(output_path, total_label_list)      # 删除无labels的图片
    
    print('Start delete empty bbox annotations:')
    delete_empty_ann(output_path, total_label_list)      # 删除无labels的anntation
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cleaning_5_labels.py')
    parser.add_argument('--set', default=r'/home/leidi/Dataset/ccpd_test',
                        type=str, help='dataset path')
    parser.add_argument('--out', default=r'/home/leidi/Dataset/ccpd_test_analyze',
                        type=str, help='output path')
    parser.add_argument('--ilstyle', '--is', dest='ilstyle', default=r'ccpd',
                        type=str, help='input labels style: ldp, hy, myxb, nuscenes, \
                                                            pascal, hy_highway, coco2017, \
                                                            kitti, cctsdb, lisa, \
                                                            hanhe，yolov5_detect, yolo, \
                                                            sjt, ccpd')
    parser.add_argument('--olstyle', '--os', dest='olstyle', default=r'ldp',
                        type=str, help='output labels style: ldp, pascal')
    parser.add_argument('--mod', default=r'yolo',
                        type=str, help='cnn model: yolo,yolo_2')
    opt = parser.parse_args()

    input_path = opt.set
    output_path = check_output_path(opt.out)
    input_label_style = opt.ilstyle
    output_label_style = opt.olstyle
    class_path = get_names_list_path(output_path)
    model = opt.mod

    print('\nStart creat labels for moder: %s' % model)
    creat_label(output_path,
                input_label_style, output_label_style, model, class_path, [])
    print('Creat label done!')
