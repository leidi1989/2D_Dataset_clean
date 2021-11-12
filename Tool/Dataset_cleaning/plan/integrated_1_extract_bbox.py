'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-21 10:02:04
LastEditors: Leidi
LastEditTime: 2021-04-23 00:30:16
'''
# -*- coding: utf-8 -*-
import argparse
import time

from utils.utils import *
from utils.extract_function import *
from utils.extract_bbox_function import *


def extract_bbox(input_path, src_set_style, pick_classes_path, out_set_style, save_image):
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
    output_image_style = check_image_type(out_set_style)    # 输出剪切图片格式
    pickclasses_list = get_class(pick_classes_path)      # 读取挑选类别列表

    data_list = []      # 声明数据集信息列表
    data_list = pickup_data_from_function(
        src_set_style, input_path, pickclasses_list)    # 抽取源标签图像数据信息
    # 依据抽取的data_list信息截取源图片bbox，并将bbox信息存为cut_image_list
    cut_image_list = creat_bbox_image(
        data_list, output_image_style, output_image_path, pickclasses_list)
    n = len(cut_image_list)
    print('Cut bbox end, total images:{0}'.format(n))

    if save_image == 1:
        print('Start save bbox images to: {0}'.format(output_image_path))
        n = save_cut_bbox_image(cut_image_list, output_image_path)
        print('Save cut bbox images, total output images:{0}'.format(n))
        
    return cut_image_list


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='extract_bbox.py')
    parser.add_argument('--set', default=r'/media/leidi/My Passport/data/hy_dataset/CCTSDB_analyze',
                        type=str, help='dataset path')
    parser.add_argument('--istyle', '--is', dest='istyle', default=r'ldp',
                        type=str, help='source detaset style:ldp')
    parser.add_argument('--ostyle', '--os', dest='ostyle', default=r'ldp',
                        type=str, help='output detaset style: ldp, yolov5')
    parser.add_argument('--pickclasses', '--pc', dest='pickclasses',
                        default=r'/media/leidi/My Passport/data/hy_dataset/CCTSDB_analyze/ImageSets/classes.txt',
                        type=str, help='dataset class path')
    parser.add_argument('--saveimg', '--si', dest='saveimg', default=1,
                        type=str, help='save cut images.')
    opt = parser.parse_args()
    
    input_path = check_input_path(opt.set)
    pick_classes_path = check_input_path(opt.pickclasses)
    src_set_style = opt.istyle
    out_set_style = opt.ostyle
    save_image = opt.saveimg

    print('Start cut bbox, and output images:')
    cut_image_list = extract_bbox(input_path, src_set_style, pick_classes_path,
                              out_set_style, save_image)
    
    time_end = time.time()
    print('time cost', time_end - time_start, 's')