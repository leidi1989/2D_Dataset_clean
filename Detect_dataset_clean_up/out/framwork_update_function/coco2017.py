'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-10-29 10:36:13
'''
import os
import shutil


def copy_images(dataset: dict, file_name: str, process_output, images_output_path: str) -> None:
    """[复制图片、数据至输出文件夹]

    Args:
        dataset (dict): [数据集信息字典]
        file_name (str): [文件名]
        process_output (dict): [进程通信字典]
        images_output_path (str): [图片输出路径]
        annotations_output_path (str): [标签输出路径]
    """

    image_path = os.path.join(
        dataset['temp_images_folder'], file_name + '.' + dataset['target_image_form'])
    image_path_output_path = os.path.join(
        images_output_path, file_name + '.' + dataset['target_image_form'])
    shutil.copy(image_path, image_path_output_path)
    process_output['image_count'] += 1
    
    return
