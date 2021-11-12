'''
Description: 
Version: 
Author: Leidi
Date: 2021-05-16 00:10:35
LastEditors: Leidi
LastEditTime: 2021-05-16 01:58:26
'''
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
from tqdm import tqdm

import sys
sys.path.append('/home/leidi/Workspace/2D_Dataset_clean/Dataset_cleaning')
from utils.utils import *
from utils.extract_function import *
import cleaning_5_labels


def dataset_list_fusion(data_list_1, data_list_2):
    """[融合两个数据集data_list列表]

    Args:
        data_list_1 ([list]): [数据集1的data_list列表]
        data_list_2 ([list]): [数据集2的data_list列表]

    Returns:
        data_list[list]: [两个数据集data_list列表]
    """    
    data_list = []
    
    data_dict_1 = {}
    print('开始进行主数据集字典转换：')
    for one in tqdm(data_list_1):
        data_dict_1[one.image_name] = one
        
    data_dict_2 = {}
    print('开始进行融合数据集字典转换：')
    for one in tqdm(data_list_2):
        data_dict_2[one.image_name] = one
        
    print('开始进行数据集融合：')
    for key, value in tqdm(data_dict_1.items()):
        try:
            value.true_box_list = data_dict_1[key].true_box_list + data_dict_2[key].true_box_list
            data_list.append(value)
        except:
            continue
    return data_list

def extract_2_dataset(input_path, input_path_fusion, output_path, input_label_style, input_label_style_fusion,
                      output_label_style, class_list, class_list_fusion, copy):
    """[summary]

    Args:
        input_path ([str]): [输入主数据集路径]]
        input_path_fusion ([str]): [输入待融合数据集路径]
        output_path ([str]): [输出数据集路径]
        input_label_style ([str]): [输入主数据集格式]
        input_label_style_fusion ([str]): [输入待融合数据集格式]
        output_label_style ([str]): [输出数据集类别]
        class_list ([list]): [输入主数据集类别列表]
        class_list_fusion ([list]): [输入待融合数据集类别列表]

    Returns:
        data_list[list]: [融合后数据列表]
    """

    source_labels_input_path_1 = input_path
    source_labels_input_path_2 = input_path_fusion
    output_path = check_output_path(output_path)
    output_path_ann = check_output_path(output_path, 'Annotations')
    total_class_list = class_list + class_list_fusion

    with open(os.path.join(output_path, 'classes.names'), 'w', encoding='utf-8') as class_names_file:    # 重写修改后的classes.names文件
        class_names_file.write("\n".join(str(x) for x in total_class_list))
    
    if 1 == copy:
        print('\n开始复制图片：')
        image_path = os.path.join(source_labels_input_path_1, 'JPEGImages')
        image_target_path = check_output_path(os.path.join(output_path, 'JPEGImages'))
        for root, dirs, files in tqdm(os.walk(image_path)):
            for file in tqdm(files):
                src_file = os.path.join(root, file)
                shutil.copy(src_file, image_target_path)

        print('\n开始复制主数据集source label：')
        source_label_path_1 = os.path.join(
            source_labels_input_path_1, 'source_label')
        source_label_target_path_1 = check_output_path(
            os.path.join(output_path, 'source_label', 'source_label 1'))
        for root, dirs, files in tqdm(os.walk(source_label_path_1)):
            for file in tqdm(files):
                src_file = os.path.join(root, file)
                shutil.copy(src_file, source_label_target_path_1)

        print('\n开始复制融合数据集source label：')
        source_label_path_2 = os.path.join(
            source_labels_input_path_2, 'source_label')
        source_label_target_path_2 = check_output_path(
            os.path.join(output_path, 'source_label', 'source_label 2'))
        for root, dirs, files in tqdm(os.walk(source_label_path_2)):
            for file in tqdm(files):
                src_file = os.path.join(root, file)
                shutil.copy(src_file, source_label_target_path_2)

    print('\n开始提取数据集信息：')
    data_list_1 = pickup_data_from_function(input_label_style,
                                            source_labels_input_path_1, class_list)
    data_list_2 = pickup_data_from_function(input_label_style_fusion,
                                            source_labels_input_path_2, class_list_fusion)
    data_list = dataset_list_fusion(data_list_1, data_list_2)
    
    pickup_data_out_function(output_label_style, output_path_ann, data_list)

    return data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='tool_label_fusion.py')
    parser.add_argument('--set', default=r'/media/leidi/MyPassport/data/hy_dataset/CCTSDB_7_classes_detect_voc',
                        type=str, help='dataset path')
    parser.add_argument('--setfusion', default=r'/media/leidi/MyPassport/data/hy_dataset/CCTSDB_analyze',
                        type=str, help='dataset path')
    parser.add_argument('--out', default=r'/media/leidi/MyPassport/data/hy_dataset/CCTSDB_10_classes_detect_voc',
                        type=str, help='output path')
    parser.add_argument('--ilstyle', '--is', dest='ilstyle', default=r'ldp',
                        type=str, help='input labels style: ldp, hy, myxb, nuscenes, \
                                                            voc, kitti, cctsdb, lisa, \
                                                            hanhe，sjt， yolov5_detect')
    parser.add_argument('--ilstylefusion', '--isf', dest='ilstylefusion', default=r'ldp',
                        type=str, help='input labels style: ldp, hy, myxb, nuscenes, \
                                                            voc, kitti, cctsdb, lisa, \
                                                            hanhe，sjt， yolov5_detect')
    parser.add_argument('--olstyle', '--os', dest='olstyle', default=r'ldp',
                        type=str, help='output labels style: ldp, pascal')
    parser.add_argument('--mod', default=r'yolo',
                        type=str, help='output labels style for cnn model: yolo')
    parser.add_argument('--oneclass', '--ol', dest='oneclass', default=None,
                        type=str, help='creat labels')
    parser.add_argument('--label', dest='label', default=1,
                        type=str, help='creat labels')
    parser.add_argument('--copy', dest='copy', default=0,
                        type=str, help='copy: JPEGImages, source_label')
    opt = parser.parse_args()

    input_path = opt.set
    input_path_fusion = opt.setfusion
    output_path = check_output_path(opt.out)
    input_label_style = opt.ilstyle
    input_label_style_fusion = opt.ilstylefusion
    output_label_style = opt.olstyle
    model = opt.mod
    label = opt.label
    one_class = opt.oneclass
    copy = opt.copy

    class_path = get_names_list_path(input_path)
    class_list = get_class(get_names_list_path(input_path))     # 获取类别列表

    class_path_fusion = get_names_list_path(input_path_fusion)
    class_list_fusion = get_class(
        get_names_list_path(input_path_fusion))     # 获取类别列表

    data_list_return = extract_2_dataset(input_path, input_path_fusion, output_path, input_label_style, input_label_style_fusion,
                                         output_label_style, class_list, class_list_fusion, copy)     # 获取源labels数据
    print('Create each image xml file done!\n')

    if None != one_class:
        data_list_return = delete_other_class(
            data_list_return, one_class)     # 删除非指定类别图片信息

    class_total_path = class_path = get_names_list_path(output_path)
    if 1 == label:
        print('\nStart xml change to moder %s:' % model)
        cleaning_5_labels.creat_label(output_path, input_label_style,
                                      output_label_style, model, class_total_path, data_list_return)     # 输出转换后的labels
        print('Creat label done!')
