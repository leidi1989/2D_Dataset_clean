'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2021-10-28 14:44:38
'''
import os
import cv2

from utils.utils import *
from base.image_base import *
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_box_list


def load_image_base_information(dataset: dict, image_base_information: dict, total_annotations_dict: dict) -> None:
    """[读取标签获取图片基础信息，并添加至each_annotation_images_data_dict]

    Args:
        dataset (dict): [数据集信息字典]
        one_image_base_information (dict): [单个数据字典信息]
        each_annotation_images_data_dict进程间通信字典 (dict): [each_annotation_images_data_dict进程间通信字典]
    """

    one_line_list = image_base_information.split(';')
    image_name = one_line_list[0]
    image_name_new = dataset['file_prefix'] + image_name.replace(
        '.'+dataset['source_image_form'], '.'+dataset['target_image_form'])
    if '' in one_line_list:
        return
    if not(image_name_new in total_annotations_dict.keys()):
        image_path = os.path.join(
            dataset['temp_images_folder'], image_name_new)
        img = cv2.imread(image_path)
        if img is None:
            print('Can not load: {}'.format(image_name_new))
            return
        height, width, channels = img.shape     # 读取每张图片的shape
        image = IMAGE(image_name, image_name_new,
                      image_path, height, width, channels, [])
    else:
        return
    total_annotations_dict.update({image_name_new: image})

    return


def load_image_annotation(dataset: dict, one_annotation: str, total_annotations_dict: dict) -> list:
    """[读取单个标签详细信息，并添加至each_annotation_images_data_dict]

    Args:
        dataset (dict): [数据集信息字典]
        one_annotation (dict): [单个数据字典信息]
        total_annotations_dict (dict): [进程间通信字典]

    Returns:
        list: [ann_image_id, true_box_list]
    """

    one_line_list = one_annotation.split(';')
    image_name = one_line_list[0]
    image_name_new = dataset['file_prefix'] + image_name.replace(
        '.'+dataset['source_image_form'], '.'+dataset['target_image_form'])
    if '' in one_line_list:
        return
    if not(image_name_new in total_annotations_dict.keys()):
        return
    else:
        # 获取真实框信息
        true_box_list = []
        cls = str(one_line_list[5])
        cls = cls.strip(' ').lower()
        if cls not in dataset['source_class_list']:
            return
        xmin = min(max(min(float(one_line_list[1]), float(
            one_line_list[3])), 0.), float(total_annotations_dict[image_name_new].width))
        ymin = min(max(min(float(one_line_list[4]), float(
            one_line_list[2])), 0.), float(total_annotations_dict[image_name_new].height))
        xmax = max(min(max(float(one_line_list[3]), float(
            one_line_list[1])), float(total_annotations_dict[image_name_new].width)), 0.)
        ymax = max(min(max(float(one_line_list[2]), float(
            one_line_list[4])), float(total_annotations_dict[image_name_new].height)), 0.)
        true_box_list.append(TRUE_BOX(
            cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表

    return image_name_new, true_box_list


def output_temp_annotation(dataset: dict, image: IMAGE, process_output: dict) -> None:
    """[输出单个标签详细信息至temp annotation]

    Args:
        dataset (dict): [数据集信息字典]
        image (IMAGE): [IMAGE类实例]
        process_output (dict): [进程间计数通信字典]
    """

    if image == None:
        return
    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        image.file_name_new + '.' + dataset['temp_annotation_form'])
    modify_true_box_list(image, dataset['modify_class_dict'])
    if dataset['class_pixel_distance_dict'] is not None:
        class_box_pixel_limit(dataset, image.true_box_list)
    if 0 == len(image.true_box_list):
        print('{} no true segmentation, has been delete.'.format(
            image.image_name_new))
        os.remove(image.image_path)
        process_output['no_true_box_count'] += 1
        process_output['fail_count'] += 1
        return
    if TEMP_OUTPUT(temp_annotation_output_path, image):
        process_output['temp_file_name_list'].append(image.file_name_new)
        process_output['success_count'] += 1
    else:
        process_output['fail_count'] += 1

    return
