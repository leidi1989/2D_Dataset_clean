'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-12-21 16:12:59
'''
import os
import shutil


def load_annotation(dataset: dict, source_annotation_name: str, process_output) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        source_annotation_path (str): [源标签路径]
        process_output ([dict]): [进程通信字典]
    """
    
    source_annotation_path = os.path.join(
        dataset['source_annotations_folder'], source_annotation_name)
    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        dataset['file_prefix'] + source_annotation_name)

    shutil.copy(source_annotation_path, temp_annotation_output_path)

    image_name = source_annotation_name.split(
        os.sep)[-1].replace('.json', '.png')
    image_name_new = dataset['file_prefix'] + image_name
    file_name_new = os.path.splitext(image_name_new)[0]
    process_output['temp_file_name_list'].append(file_name_new)
    process_output['success_count'] += 1

    return
