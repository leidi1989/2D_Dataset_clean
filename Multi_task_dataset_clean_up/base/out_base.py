'''
Description:
Version:
Author: Leidi
Date: 2021-08-03 22:18:39
LastEditors: Leidi
LastEditTime: 2021-09-14 18:04:42
'''
from utils.utils import temp_annotation_path_list
from annotation.annotation_output import annotation_output_funciton


def out(dataset: dict) -> None:
    """[输出target annotation]

    Args:
        dataset (dict): [数据集信息字典]
    """    
    dataset['temp_annotation_path_list'] = temp_annotation_path_list(
        dataset['temp_annotations_folder'])
    annotation_output_funciton(dataset['target_dataset_style'], dataset)
    
    return
