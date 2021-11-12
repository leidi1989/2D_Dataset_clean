'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-21 10:02:04
LastEditors: Leidi
LastEditTime: 2021-04-21 13:48:01
'''
# -*- coding: utf-8 -*-
import argparse
import time

from utils.utils import *
from utils.extract_function import *
# from utils.fix_function import *
# from utils.label import *


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='image_integrated.py')
    parser.add_argument('--set', default=r'E:\2.Datasets\hy_mix_dataset_yolo_input_analyze',
                        type=str, help='dataset path')
    parser.add_argument('--srcnames', '--sn', dest='srcnames',
                        default=r'E:\2.Datasets\hy_mix_dataset_yolo_input_analyze\ImageSets\classes.names',
                        type=str, help='dataset class path')
    parser.add_argument('--fixnamesfile', '--fn', dest='fixnamesfile',
                        default=r'E:\2.Datasets\dataset_mix\hy_coco_fix_class_file\fix_classes.txt',
                        type=str, help='dataset class path')
    parser.add_argument('--out', default=r'E:\2.Datasets\hy_self_mix_dataset',
                        type=str, help='output path')
    parser.add_argument('--istyle', '--is', dest='istyle', default=r'hy',
                        type=str, help='source detaset style: ldp, hy')
    parser.add_argument('--ostyle', '--os', dest='ostyle', default=r'hy',
                        type=str, help='output detaset style: ldp, hy, yolov5')
    opt = parser.parse_args()

    input_path = check_input_path(opt.set)
    output_path = check_output_path(opt.out)
    src_class_path = opt.srcnames
    fix_class_file_path = opt.fixnamesfile
    src_set_style = opt.istyle
    out_set_style = opt.ostyle

    # src_set_fold_list = pickup_fold_function(
    #     src_set_style)     # 获取对应数据集类型的文件路径
    # for one in src_set_fold_list:       # 创建融合数据集目录
    #     check_output_path(os.path.join(output_path, one))
    
    # src_put_path_dict = pickup_src_set_output_function(
    #     src_set_style, input_path)     # 获取源数据集下文件组织路径
    # output_path_dict = pickup_new_set_output_function(
    #     out_set_style, output_path)   # 获取新数据集下文件组织路径
    
    # total_mix_images_path_list = []     # 声明混合数据集图片路径列表

    # total_label_list_path = check_input_path(os.path.join(
    #     input_path, src_set_fold_list[2], 'total.txt'))   # 获取源数据集total.txt
    # image_type = check_image_type(src_set_style)    # 获取源数据集类别对应图片类别
    # classes_src_list = get_class(src_class_path)    # 获取源数据集类别
    # classes_fix_dict = get_fix_class_dict(fix_class_file_path)    # 获取修改后类别字典
    # new_class_names_list = get_new_class_names_list(
    #     classes_fix_dict)   # 获取融合后的类别列表
    # src_total_label_path_list = get_src_total_label_path_list(
    #     total_label_list_path, image_type)    # 获取数据集全部标签列表

    # print('\nStart to dataset mix:')
    # print('\nStart to fix class in label:')
    # fix_total_label_path_dict = extract_label(
    #     src_total_label_path_list, classes_src_list, classes_fix_dict, new_class_names_list)
    # # 获取全部标签名称列表，修改标签后key为标签名称value为对应路径的全部标签字典

    # print('\nStart to creat mix labels:')
    # total_mix_images_name_list = label(
    #     fix_total_label_path_dict, new_class_names_list, src_set_style, output_path_dict)

    # print('\nStart to creat ImageSets fold file:')
    # imagesets_flod(output_path_dict, total_mix_images_name_list, src_set_style, new_class_names_list)
    # time_end = time.time()
    # print('time cost', time_end - time_start, 's')
