'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:03
LastEditors: Leidi
LastEditTime: 2022-01-25 16:33:19
'''
# -*- coding: utf-8 -*-
import os
from types import FunctionType
from PIL import Image


def png_jpg(image_path: str, image_output_path: str) -> int:
    """[将png格式图片转换为jpg格式图片]

    Args:
        image_path (str): [输入图片路径]
        image_output_path (str): [输出图片路径]

    Returns:
        int: [description]
    """
    if os.path.splitext(image_path)[-1].replace('.', '') != 'png':
        return 0
    image_output_path = image_output_path.replace('.png', '.jpg')
    img = Image.open(image_path)
    try:
        if len(img.split()) == 4:
            r, g, b, _ = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(image_output_path, quality=100)
            return 1
        else:
            img.convert('RGB').save(image_output_path, quality=100)
            return 1
    except Exception as e:
        print("PNG转换JPG 错误", e)
        return 0


def function_miss(*args) -> None:
    """提示添加缺失的图片格式转换函数"""
    print("\n无图片格式转换函数，请添加转换函数。")


image_transform_function_dict = {'png_jpg': png_jpg,
                                 }


def image_transform_function(transform_type: str, *args):
    """[根据输入类别挑选数据集图片类型转换函数]

    Args:
        transform_type (str): [转换类型]

    Returns:
        [function]: [转换函数]
    """

    return image_transform_function_dict.get(transform_type, function_miss)(*args)
