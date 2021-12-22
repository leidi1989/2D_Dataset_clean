'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-12-22 15:58:51
'''
import shutil

from annotation.annotation_temp import TEMP_LOAD


def copy_image(x: str, image_output_path: str) -> None:
    """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀]

    Args:
        dataset (dict): [数据集信息字典]
        root (str): [文件所在目录]
        n (str): [文件名]

    """

    shutil.copy(x, image_output_path)
    return