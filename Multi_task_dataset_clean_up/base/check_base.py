'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-19 14:12:46
LastEditors: Leidi
LastEditTime: 2021-12-22 13:44:02
'''
import shutil

from utils.utils import *
from annotation import annotation_check
from utils.plot import plot_true_segment, plot_true_box


def check(dataset: dict) -> None:
    """[进行标签检测]

    Args:
        dataset (dict): [数据集信息字典]
    """

    dataset['check_images_list'] = annotation_check.__dict__[
        dataset['target_dataset_style']](dataset)
    shutil.rmtree(dataset['check_annotation_output_folder'])
    check_output_path(dataset['check_annotation_output_folder'])
    plot_true_box(dataset)
    plot_true_segment(dataset)

    return
