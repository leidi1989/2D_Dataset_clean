'''
Description:
Version:
Author: Leidi
Date: 2021-08-03 22:18:39
LastEditors: Leidi
LastEditTime: 2021-12-31 14:13:19
'''
import shutil

from utils.utils import *
from annotation import annotation_check
from utils.plot import plot_true_box, plot_true_segmentation


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
    plot_true_segmentation(dataset)

    return
