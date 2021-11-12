'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-19 20:56:09
LastEditors: Leidi
LastEditTime: 2021-10-21 19:47:09
'''
import os
import shutil


def annotation_output(dataset: dict, temp_annotation_path: str) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        temp_annotation_path (str): [暂存标签路径]
    """

    file = temp_annotation_path.split(os.sep)[-1].split('.')[0]
    annotation_output_path = annotation_output_path = os.path.join(
        dataset['target_annotations_folder'], file + '.' + dataset['target_annotation_form'])
    shutil.copy(temp_annotation_path, annotation_output_path)

    return
