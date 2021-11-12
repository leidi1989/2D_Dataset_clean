'''
Description:
Version:
Author: Leidi
Date: 2021-08-03 22:18:39
LastEditors: Leidi
LastEditTime: 2021-09-27 18:19:54
'''
from utils.utils import *
from utils.plot import plot_true_segment, plot_true_box
from annotation.annotation_output import *
from annotation.annotation_check import annotation_check_function

import shutil
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='AR PL UMing CN')
plt.switch_backend('agg')


def check(dataset: dict) -> None:
    """[进行标签检测]

    Args:
        dataset (dict): [数据集信息字典]
    """   
    
    dataset['check_images_list'] = image_list(dataset)
    shutil.rmtree(dataset['check_annotation_output_folder'])
    check_output_path(dataset['check_annotation_output_folder'])
    plot_true_box(dataset)
    plot_true_segment(dataset)
    
    return


def image_list(dataset: dict) -> list:
    """[获取检测标签的IMAGE实例列表]

    Args:
        dataset (dict): [数据集信息字典]

    Returns:
        list: [获取检测标签的IMAGE实例列表]
    """    
    
    return annotation_check_function(dataset['target_dataset_style'], dataset)
