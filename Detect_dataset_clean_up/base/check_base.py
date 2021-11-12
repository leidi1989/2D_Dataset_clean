'''
Description:
Version:
Author: Leidi
Date: 2021-08-03 22:18:39
LastEditors: Leidi
LastEditTime: 2021-10-22 16:23:48
'''
import shutil

from utils.utils import *
from utils.plot import plot_true_box
from annotation.annotation_check import annotation_check_function


def check(dataset: dict) -> None:
    """[进行标签检测]
    """
    dataset['check_images_list'] = image_list(dataset)
    shutil.rmtree(dataset['check_annotation_output_folder'])
    check_output_path(dataset['check_annotation_output_folder'])
    plot_true_box(dataset)

    return


def image_list(dataset: dict) -> list:
    """[获取检测标签的IMAGE实例列表]

    Returns:
        list: [IMAGE实例列表]
    """
    return annotation_check_function(dataset['target_dataset_style'], dataset)
