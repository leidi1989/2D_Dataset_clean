'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:03
LastEditors: Leidi
LastEditTime: 2021-12-22 14:54:07
'''
# -*- coding: utf-8 -*-
from PIL import Image


def png_jpg(image_path: str, image_output_path: str) -> int:
    """[将png格式图片转换为jpg格式图片]

    Args:
        image_path (str): [输入图片路径]
        image_output_path (str): [输出图片路径]

    Returns:
        int: [description]
    """

    if image_path.endswith('.png'):
        image_output_path = image_output_path.replace('.png', '.jpg')
        try:
            img = Image.open(image_path)
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
    else:
        return 0


def jpg_png(image_path: str, image_output_path: str) -> int:
    """[将jpg格式图片转换为png格式图片]

    Args:
        image_path (str): [输入图片路径]
        image_output_path (str): [输出图片路径]

    Returns:
        int: [description]
    """

    if image_path.endswith('.jpg'):
        image_output_path = image_output_path.replace('.jpg', '.png')
        try:
            img = Image.open(image_path)
            img.convert('RGB').save(image_output_path, quality=100)
            return 1
        except Exception as e:
            print("JPG转换PNG 错误", e)
            return 0
    else:
        return 0
