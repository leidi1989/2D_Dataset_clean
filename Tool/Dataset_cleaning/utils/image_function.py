'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:03
LastEditors: Leidi
LastEditTime: 2021-10-21 20:09:11
'''
# -*- coding: utf-8 -*-
import os
from PIL import Image
from tqdm import tqdm

from utils.utils import *


def from_ldp_image_dataset(root, fileName):
    """[数据集图片更名]

    Parameters
    ----------
    root : [str]
        [文件夹所在地址]
    fileName : [str]
        [文件名]

    Returns
    -------
    fileName : [str]
        [文件名]
    """    

    return fileName


def from_lisa_image_dataset(root, fileName):
    """[数据集图片更名]

    Parameters
    ----------
    root : [str]
        [文件夹所在地址]
    fileName : [str]
        [文件名]

    Returns
    -------
    fileName : [str]
        [修改后的文件名]
    """
    fileName = fileName.replace('--', '_')

    return fileName


def in_func_None(*args):
    """images如无对应输入图片函数，提示用户添加输入图片函数"""
    print("\n无对应输入图片函数，请添加输入图片函数。")
    return 0


in_iamge_func_dict = {"ldp": from_ldp_image_dataset,
                      "hy": from_ldp_image_dataset, 
                      "hy_highway":from_ldp_image_dataset,
                      "sjt": from_ldp_image_dataset, 
                      "coco2017": from_ldp_image_dataset,
                      "kitti": from_ldp_image_dataset,
                      "pascal": from_ldp_image_dataset,
                      "lisa": from_lisa_image_dataset,
                      "hanhe": from_ldp_image_dataset,
                      "myxb": from_ldp_image_dataset,
                      "nuscenes": from_ldp_image_dataset,
                      "cctsdb": from_ldp_image_dataset,
                      "compcars": from_ldp_image_dataset, 
                      "yolov5_detect": from_ldp_image_dataset,
                      "ccpd": from_ldp_image_dataset,
                      "yolo":from_ldp_image_dataset,
                      "licenseplate": from_ldp_image_dataset
                      }


def pickup_image_from_function(input_label_style, *args):
    """[根据输入类别挑文件名修改换函数]

    Parameters
    ----------
    input_label_style : [str]
        [输入标签类别]

    Returns
    -------
    in_iamge_func_dict.get() : [function]
        [输出对应类别的文件名修改函数]
    """    
    # 返回对应数据获取函数
    return in_iamge_func_dict.get(input_label_style, in_func_None)(*args)


def png_to_jpg(output_path):
    """[将image文件夹内png格式图片转换为jpg格式]

    Args:
        output_path ([str]): [数据集路径]
    """    
    print('\nStart change png to jpg:')
    output_path_images = check_output_path(output_path, 'JPEGImages')
    images_count = 0
    wrong_images_count = 0
    for one_image in tqdm(os.listdir(output_path_images)):
        if one_image.split('.')[-1] != 'png':
            continue
        in_image = os.path.join(output_path_images, one_image)
        out_image = os.path.join(output_path_images, one_image.replace('.png', '.jpg'))
        img = Image.open(in_image)
        try:
            if len(img.split()) == 4:
                r, g, b, _ = img.split()
                img = Image.merge("RGB", (r, g, b))
                img.convert('RGB').save(out_image, quality=100)
                images_count += 1
                os.remove(in_image)
            else:
                img.convert('RGB').save(out_image, quality=100)
                images_count += 1
                os.remove(in_image)
        except Exception as e:
            wrong_images_count += 1
            print("PNG转换JPG 错误", e)
            continue
    print("Total pictures: %d images have been change." % images_count)
    print("Total pictures: %d images wrong." % wrong_images_count)
    
    
def check_image_rename(input_label_style, rename_file):
    """[修改图片分隔符]

    Args:
        input_label_style ([str]): [数据集类别]
        rename_file ([str]): [图片名称]

    Returns:
        [str]: [图片名称]
    """    
    if input_label_style == 'ccpd':
        rename_file=rename_file.replace('&', '#')
    
    return rename_file