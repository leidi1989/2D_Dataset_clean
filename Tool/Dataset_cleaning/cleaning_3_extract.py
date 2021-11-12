'''
Description: 
Version: 
Author: Leidi
Date: 2020-09-04 13:21:36
LastEditors: Leidi
LastEditTime: 2021-10-21 20:08:52
'''
# -*- coding: utf-8 -*-
import argparse

from utils.utils import *
from utils.extract_function import *
import cleaning_5_labels


def extract(output_path, input_label_style, output_label_style, class_list, one_class):
    """[将源标签转换为指定类型标签]

    Parameters
    ----------
    output_path : [str]
        [输出路径]
    input_label_style : [str]
        [源数据集标签类别]
    output_label_style : [str]
        [输出数据集标签类别]
    class_list : [list]
        [类别列表]

    Returns
    -------
    data_list : [list]
        [全部数据信息列表]
    """

    source_labels_input_path = output_path
    output_path_ann = check_output_path(output_path, 'Annotations')

    data_list = []
    data_list = pickup_data_from_function(input_label_style,
                                          source_labels_input_path, class_list)
    pickup_data_out_function(output_label_style, output_path_ann, data_list)

    return data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cleaning_3_extract.py')
    parser.add_argument('--set', default=r'/home/leidi/Dataset/hy_highway_myxb_sjt_coco2017_7_classes_input_20210805',
                        type=str, help='dataset path')
    parser.add_argument('--out', default=r'/home/leidi/Dataset/hy_highway_myxb_sjt_coco2017_7_classes_output_20210805',
                        type=str, help='output path')
    parser.add_argument('--ilstyle', '--is', dest='ilstyle', default=r'ldp',
                        type=str, help='input labels style: ldp, hy, myxb, nuscenes, \
                                                            pascal, hy_highway, coco2017, \
                                                            kitti, cctsdb, lisa, \
                                                            hanhe，yolov5_detect, yolo, \
                                                            sjt, ccpd')
    parser.add_argument('--olstyle', '--os', dest='olstyle', default=r'ldp',
                        type=str, help='output labels style: ldp, pascal')
    parser.add_argument('--mod', default=r'yolo',
                        type=str, help='output labels style for cnn model: yolo')
    parser.add_argument('--oneclass', '--ol', dest='oneclass', default=None,
                        type=str, help='creat labels')
    parser.add_argument('--label', dest='label', default=1,
                        type=str, help='creat labels')
    opt = parser.parse_args()

    input_path = opt.set
    output_path = check_output_path(opt.out)
    input_label_style = opt.ilstyle
    output_label_style = opt.olstyle
    model = opt.mod
    label = opt.label
    one_class = opt.oneclass

    class_path = get_names_list_path(input_path)
    class_list = get_class(get_names_list_path(input_path))     # 获取类别列表

    data_list_return = extract(
        output_path, input_label_style, output_label_style, class_list, one_class)     # 获取源labels数据
    print('Create each image xml file done!\n')

    if None != one_class:
        data_list_return = delete_other_class(
            data_list_return, one_class)     # 删除非指定类别图片信息

    if 1 == label:
        print('\nStart xml change to moder %s:' % model)
        cleaning_5_labels.creat_label(output_path, input_label_style,
                                      output_label_style, model, class_path, data_list_return)     # 输出转换后的labels
        print('Creat label done!')
