'''
Description: 
Version: 
Author: Leidi
Date: 2021-11-08 10:33:42
LastEditors: Leidi
LastEditTime: 2021-11-08 11:45:14
'''
import os
import json
import shutil

from utils.image_form_transform import image_transform_function


def copy_image(dataset: dict, root: str, n: str) -> None:
    """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀]

    Args:
        dataset (dict): [数据集信息字典]
        root (str): [文件所在目录]
        n (str): [文件名]

    """

    image = os.path.join(root, n)
    temp_image = os.path.join(
        dataset['source_images_folder'], dataset['file_prefix'] + n)
    if dataset['source_image_form'] != dataset['target_image_form']:
        dataset['transform_type'] = dataset['source_image_form'] + \
            '_' + dataset['target_image_form']
        image_transform_function(
            dataset['transform_type'], image, temp_image)
        return
    else:
        shutil.copy(image, temp_image)
        return


def copy_annotation(dataset: dict, root: str, n: str) -> None:
    """[复制源数据集标签文件至目标数据集中的source_annotations中]

    Args:
        dataset (dict): [数据集信息字典]
        root (str): [文件所在目录]
        n (str): [文件名]
    """

    fake_js = {}
    json_name = n.replace('.jpg', '.json')
    json_output_path = os.path.join(
        dataset['source_annotations_folder'], json_name)
    json.dump(fake_js, open(json_output_path, 'w'))

    return