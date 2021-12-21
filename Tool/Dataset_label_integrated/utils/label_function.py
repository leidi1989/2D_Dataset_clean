'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-19 14:12:46
LastEditors: Leidi
LastEditTime: 2021-12-21 16:20:49
'''
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os.path import join
from tqdm import tqdm
from utils.utils import *
from utils.extract_function import *
from utils.convertion_function import *


def to_yolo(output_path, data_list, class_list):
    """将坐标转换为yolo格式，存储为txt并输出至labels"""

    total_label_list = []
    for one_image in tqdm(data_list):
        if one_image == None:
            continue
        one_image_bbox = []     # 声明每张图片bbox列表
        for bbox in one_image.true_box_list:    # 遍历单张图片全部bbox
            cls_re = str(bbox.clss)      # 获取bbox类别
            cls_re = cls_re.replace(' ', '').lower()
            if cls_re in set(class_list):
                cls_id = class_list.index(cls_re)
                b = (bbox.xmin, bbox.xmax, bbox.ymin,
                     bbox.ymax,)   # 获取源标签bbox的xxyy
                # 转换bbox至yolo类型
                bb = yolo((one_image.width, one_image.height), b)
                one_image_bbox.append([cls_id, bb])
            else:
                print('\nErro! class not in classes.names image: %s!' % one_image.image_name)
        image_name = os.path.splitext(one_image.image_name)[0]     # 获取图片名称
        with open(os.path.join(output_path, image_name + '.txt'), 'w') as one_image_label:   # 创建图片对应txt格式的label文件
            for one_bbox in one_image_bbox:
                one_image_label.write(str(one_bbox[0]) + " " +
                                      " ".join([str(a) for a in one_bbox[1]]) + '\n')
        total_label_list.append(image_name)

    return total_label_list


def yolo_to_yolo(output_path, data_list, class_list):
    """将坐标转换为yolo格式，存储为txt并输出至labels"""

    total_label_list = []
    for one_image in tqdm(data_list):
        if one_image == None:
            continue
        one_image_bbox = []     # 声明每张图片bbox列表
        for bbox in one_image.true_box_list:    # 遍历单张图片全部bbox
            cls_re = str(bbox.clss)      # 获取bbox类别
            cls_re = int(class_list.index(cls_re))
            if cls_re <= len(class_list):
                cls_id = cls_re
                b = (bbox.xmin, bbox.xmax, bbox.ymin,
                     bbox.ymax,)   # 获取源标签bbox的xxyy
                # 转换bbox至yolo类型
                bb = yolo((one_image.width, one_image.height), b)
                one_image_bbox.append([cls_id, bb])
            else:
                print('\nErro image: %s!\n' % one_image.image_name)
        image_name = os.path.splitext(one_image.image_name)[0]     # 获取图片名称
        with open(os.path.join(output_path, image_name + '.txt'), 'w') as one_image_label:   # 创建图片对应txt格式的label文件
            for one_bbox in one_image_bbox:
                one_image_label.write(str(one_bbox[0]) + " " +
                                      " ".join([str(a) for a in one_bbox[1]]) + '\n')
        total_label_list.append(image_name)

    return total_label_list

label_out_func_dict = {"yolo": to_yolo, "yolo_2":yolo_to_yolo}


def pickup_data_out_to_label_function(model, *args):
    """根据输入类别挑选转换函数"""

    # 返回对应数据输出函数
    return label_out_func_dict.get(model, out_func_None)(*args)
