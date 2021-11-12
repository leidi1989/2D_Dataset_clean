'''
Description: 
Version: 
Author: Leidi
Date: 2020-10-23 22:14:51
LastEditors: Leidi
LastEditTime: 2021-08-02 15:35:41
'''
# -*- coding: utf-8 -*-
import argparse
import os
import time

from utils.utils import *
from utils.annotation_integrated_function import *
from utils.extract_function import *


def annotation_integrated(output_path, src_class_path, fix_class_file_path, data_list):
    """[更换annotation的图片类中的true_box类别，完成标签融合]

    Parameters
    ----------
    output_path : [str]
        [数据集输入输出路径]
    src_class_path : [str]
        [源类别路径]
    fix_class_file_path : [str]
        [类别修改文件路径]
    data_list : [list]
        [annotation图片数据信息]

    Returns
    -------
    [list]
    data_list : [修改后的data_list信息]
    """    

    classes_src_list = get_class(src_class_path)    # 获取源数据集类别
    classes_fix_dict = get_fix_class_dict(fix_class_file_path)    # 获取修改后类别字典
    new_class_names_list = get_new_class_names_list(
        classes_fix_dict)   # 获取融合后的类别列表
    output_path_ann = check_output_path(
        os.path.join(output_path, "Annotations"))
    output_path_2 = check_output_path(
        output_path, 'ImageSets')     # 输出ImageSets路径

    data_list_extract = []
    if len(data_list) == 0:
        print('\nStart to extract dataset annotations:')
        data_list_extract = pickup_data_from_function(
            'ldp', output_path, classes_src_list)
    else:
        data_list_extract = data_list

    print('\nStart to fix class in annotations:')
    # 修数据集annotation中的真实框类别，进行类别融合
    data_list_extract = fix_annotation(data_list_extract, classes_fix_dict)
    
    data_list_extract = cheak_total_images_data_list(data_list_extract)   # 在总图片信息列表中剔除无真实框图片信息
    
    print('\nStart to creat mix annotations:')
    pickup_data_out_function('ldp', output_path_ann, data_list_extract)
    
    print('\nStart to fix classes.names:')
    with open(os.path.join(output_path_2, 'source_classes.names'), 'w') as class_list_out:   # 创建图片对应txt格式的源classes.names文件
        class_list_out.write('\n'.join(str(x) for x in classes_src_list))
            
    with open(os.path.join(output_path, 'classes.names'), 'w', encoding='utf-8') as class_names_file:    # 重写修改后的classes.names文件
        class_names_file.write('\n'.join(str(x) for x in new_class_names_list))
            
    return data_list_extract


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='label_integrated.py')
    parser.add_argument('--names', '--sn', dest='names',
                        default=r'/home/leidi/Dataset/ccpd_test/classes.names',
                        type=str, help='dataset class path')
    parser.add_argument('--fixclassfile', '--fcf', dest='fixclassfile',
                        default=r'/home/leidi/Dataset/1_classes_20210709.txt',
                        type=str, help='dataset class path')
    parser.add_argument('--out', default=r'/home/leidi/Dataset/ccpd_test_analyze',
                        type=str, help='output path')
    opt = parser.parse_args()

    output_path = check_output_path(opt.out)
    src_class_path = opt.names
    fix_class_file_path = opt.fixclassfile

    annotation_integrated(output_path, src_class_path, fix_class_file_path, [])

    time_end = time.time()
    print('time cost', time_end - time_start, 's')
