'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:02
LastEditors: Leidi
LastEditTime: 2021-10-21 20:09:04
'''
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time

# import sys
# root_path = os.path.normpath(os.path.abspath('/Dataset_cleaning'))
# sys.path.insert(0, root_path)

from utils.utils import *
from utils.extract_function import *
import tool.tool_move
import cleaning_1_images
import cleaning_2_keys
import cleaning_3_extract
import cleaning_4_annotations_integrated
import cleaning_5_labels
import cleaning_6_divideset
import cleaning_7_distribution
import cleaning_8_checklabels
import cleaning_9_test_labels


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='cleaning.py')
    parser.add_argument('--set', default=r'/home/leidi/Dataset/myxb_df_highway_urban_input_20210806',
                        type=str, help='dataset path')
    parser.add_argument('--names', default=r'/home/leidi/Dataset/myxb_df_highway_urban_input_20210806/classes.names',
                        type=str, help='dataset class path')
    parser.add_argument('--out', default=r'/home/leidi/Dataset/myxb_df_highway_urban_output_20210806',
                        type=str, help='output path')
    parser.add_argument('--pref', default=r'myxb',
                        type=str, help='images or labels need to rename prefix')
    parser.add_argument('--segment', '--sg', dest='segment', default='_',
                        type=str, help='name split')
    parser.add_argument('--ilstyle', '--is', dest='ilstyle', default=r'myxb',
                        type=str, help='input labels style: ldp, hy, myxb, nuscenes, \
                                                            pascal, hy_highway, coco2017, \
                                                            kitti, cctsdb, lisa, \
                                                            hanhe，yolov5_detect, yolo, \
                                                            sjt, ccpd, licenseplate')
    parser.add_argument('--olstyle', '--os', dest='olstyle', default=r'ldp',
                        type=str, help='output labels style: ldp, pascal')
    parser.add_argument('--oneclass', '--ol', dest='oneclass', default='',
                        type=str, help='creat labels')
    parser.add_argument('--fixclass', '--fc', dest='fixclass', default=0,
                        type=str, help='fix dataset class')
    parser.add_argument('--fixclassfile', '--fcf', dest='fixclassfile',
                        default=r'/mnt/disk1/dataset/0.dataset_labels_integrated_dict/licenseplate_code_65_classes_20210802.txt',
                        type=str, help='dataset class path')
    parser.add_argument('--mod', default=r'yolo',
                        type=str, help='output labels style for cnn model: yolo, yolo_2, test_yolo')
    parser.add_argument('--ratio', '--ra', dest='ratio', default=(0.8, 0.1, 0.1, 0),
                        nargs=3, type=float, help='(train, val, test, redund)')
    parser.add_argument('--check', dest='check', default=1000,
                        type=int, help='check labels')
    parser.add_argument('--mask', dest='mask', default=0,
                        type=str, help='transparent bounding box mask')
    parser.add_argument('--testlabel', '--tl', dest='testlabel', default=0,
                        type=str, help='fix dataset class')
    parser.add_argument('--tov5', dest='tov5', default=0,
                        type=str, help='fix dataset to yolo_v5')
    parser.add_argument('--nout', default=r'',
                        type=str, help='new output path')
    parser.add_argument('--detc', '--d', dest='detc', default=0,
                        type=str, help='output detaset style: yolov5')
    opt = parser.parse_args()

    input_path = check_input_path(opt.set)
    output_path = check_output_path(opt.out)
    output_label_style = opt.olstyle
    model = opt.mod
    ratio = opt.ratio
    mask = opt.mask
    input_label_style = opt.ilstyle
    check = opt.check
    fixclass = opt.fixclass
    fix_class_file_path = opt.fixclassfile
    test_label = opt.testlabel
    one_class = opt.oneclass
    tov5 = opt.tov5
    nout = opt.nout
    for_detect = opt.detc
    segment = opt.segment
    name_pre = check_pref(opt.pref, segment)

    class_path = get_names_list_path(input_path)
    classes_names_output_path = os.path.join(output_path, 'classes.names')
    if os.path.exists(classes_names_output_path):
        os.remove(classes_names_output_path)
    shutil.copyfile(class_path, classes_names_output_path)

    output_class_path = get_names_list_path(output_path)
    class_list = get_class(get_names_list_path(input_path))

    print('\nStart images change:')
    cleaning_1_images.images_move(
        input_path, output_path, name_pre, input_label_style)
    print('Change each image file done!')

    print('\nStart source label change:')
    cleaning_2_keys.source_label_change_key(
        input_path, output_path, name_pre, input_label_style)
    print('Change each label done!')

    print('\nStart source label change to xml:')
    data_list_return = cleaning_3_extract.extract(
        output_path, input_label_style, output_label_style, class_list, one_class)
    print('Create each image xml file done!')

    if fixclass == 1:
        data_list_return = cleaning_4_annotations_integrated.annotation_integrated(
            output_path, class_path, fix_class_file_path, data_list_return)
    print('Fix xml file\'s true box class done!')

    print('\nStart xml change to moder %s:' % model)
    cleaning_5_labels.creat_label(
        output_path, input_label_style, output_label_style, model, output_class_path, data_list_return)
    print('Creat label done!')

    print('\nStart to divide dataset：')
    cleaning_6_divideset.divide_set(
        output_path, input_label_style, ratio, segment)
    print('Divide set done!')

    print('\nStart to distribution dataset：')
    cleaning_7_distribution.distribution(output_path)
    print('Dataset analyze done!')

    if check != 0:
        print('\nStart to check dataset label:')
        cleaning_8_checklabels.checklabels(
            output_path, input_label_style, mask, check)
        print('Check dataset done!')

    if test_label == 1:
        print('\nStart to create test label:')
        test_model = 'test_yolo'
        cleaning_9_test_labels.creat_test_label(
            output_path, input_label_style, output_label_style, test_model, output_class_path, data_list_return)
        print('Create test label done!')

    if 1 == tov5:
        print('Start to move dataset image and label:')
        output_path = check_output_path(opt.out)
        new_output_path = check_output_path(opt.nout)
        src_set_style = opt.ilstyle
        out_set_style = r'yolov5'

        tool.tool_move.change_datast_floder(output_path, src_set_style,
                                            out_set_style, new_output_path, for_detect)
        print('Move dataset image and label done!')

    time_end = time.time()
    print('time cost', time_end-time_start, 's')
