'''
Description:
Version:
Author: Leidi
Date: 2021-08-03 22:18:39
LastEditors: Leidi
LastEditTime: 2021-11-08 19:12:05
'''
import shutil

from utils.utils import *
from utils.plot import plot_true_box
from annotation import annotation_check


def check(dataset: dict) -> None:
    """[进行标签检测]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Start check output annotation:')
    dataset['check_images_list'] = annotation_check.__dict__[
        dataset['target_dataset_style']](dataset)
    shutil.rmtree(dataset['check_annotation_output_folder'])
    check_output_path(dataset['check_annotation_output_folder'])
    plot_true_box(dataset)

    return
