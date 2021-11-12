'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-05 16:33:40
LastEditors: Leidi
LastEditTime: 2021-08-11 18:43:42
'''
# -*- coding: utf-8 -*-
from base.image_base import IMAGE
from tqdm import tqdm

from utils.utils import *


def modify_true_box_list(image: IMAGE, class_modify_dict: dict) -> None:
    """[修改真实框类别]

    Args:
        image (IMAGE): [IMAGE类变量]
        class_modify_dict (dict): [类别修改字典]
    """
    if class_modify_dict is not None:
        for one_true_box in image.true_box_list:
            # 遍历融合类别文件字典，完成label中的类别修改，
            # 若此bbox类别属于混合标签类别列表，
            # 则返回该标签在混合类别列表的索引值
            for (key, value) in class_modify_dict.items():
                if one_true_box.clss in set(value):
                    one_true_box.clss = key                     # 修改true_box类别


def modify_annotation_class(total_images_data_list: list, classes_fix_dict: dict) -> list:
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
    for one_image in tqdm(total_images_data_list):          # 遍历源数据集标签列表
        one_image.true_box_list = modify_true_box_list(
            one_image.true_box_list, classes_fix_dict)      # 生成新真实框列表的图片类实例

    return total_images_data_list                           # 返回修改后的图片数据信息列表
