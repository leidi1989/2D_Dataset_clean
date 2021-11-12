# -*- coding: utf-8 -*-
import argparse
import os
from utils.utils import *
from utils.fix_function import *
from utils.output_path_function import *
from utils.label_output import *
from tqdm import tqdm
import time


def fix_true_box_list(true_box_list, classes_fix_dict):
    """[修改输入真实框列表中的类别，完成类别融合]

    Parameters
    ----------
    true_box_list : [list]
        [图片真实框列表]
    classes_fix_dict : [dict]
        [修改类别字典]

    Returns
    -------
    new_true_box_list : [list]
        [修改类别后的真实框列表]
    """

    new_true_box_list = []      # 声明新类别真实框列表
    for one_true_box in true_box_list:
        for (key, value) in classes_fix_dict.items():   # 遍历融合类别文件字典，完成label中的类别修改
            # 若此bbox类别属于混合标签类别列表，则返回该标签在混合类别列表的索引值
            if one_true_box.clss in set(value):
                one_true_box.clss = key     # 修改true_box类别
                new_true_box_list.append(one_true_box)

    return new_true_box_list    # 返回新真实框列表


def fix_annotation(total_images_data_list, classes_fix_dict):
    """[完成total_images_data_list中各个图片类中的true_box的类别修改]

    Parameters
    ----------
    total_images_data_list : [list]
        [数据集图片数据信息列表]
    classes_fix_dict : [type]
        [类别修改字典]

    Returns
    -------
    total_images_data_list : [list]
        [数据集图片数据信息列表]
    """    

    for one_image in tqdm(total_images_data_list):   # 遍历源数据集标签列表
        one_image.true_box_list = fix_true_box_list(
            one_image.true_box_list, classes_fix_dict)  # 生成新真实框列表的图片类实例

    return total_images_data_list   # 返回修改后的图片数据信息列表
