'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2021-10-28 14:43:00
'''
import os
import cv2
import xml.etree.ElementTree as ET

from base.image_base import *
from utils.utils import class_pixel_limit
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_box_list


def load_annotation(dataset: dict, source_annotations_name: str, process_output) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        source_annotations_name (str): [源标签名称]
        process_output ([dict]): [进程通信字典]
    """

    source_annotation_path = os.path.join(
        dataset['source_annotations_folder'], source_annotations_name)
    tree = ET.parse(source_annotation_path)
    root = tree.getroot()
    image_name = str(root.find('filename').text)
    image_name_new = dataset['file_prefix'] + \
        str(root.find('filename').text)
    image_path = os.path.join(
        dataset['temp_images_folder'], image_name_new)
    img = cv2.imread(image_path)
    if img is None:
        print('Can not load: {}'.format(image_name_new))
        return
    height, width, channels = img.shape
    true_box_dict_list = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = str(obj.find('name').text)
        cls = cls.replace(' ', '').lower()
        if cls not in dataset['source_class_list']:
            continue
        if int(difficult) == 1:
            continue
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text),
             float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        xmin = max(min(float(b[0]), float(b[1]), float(width)), 0.)
        ymin = max(min(float(b[2]), float(b[3]), float(height)), 0.)
        xmax = min(max(float(b[1]), float(b[0]), 0.), float(width))
        ymax = min(max(float(b[3]), float(b[2]), 0.), float(height))
        true_box_dict_list.append(TRUE_BOX(
            cls, xmin, ymin, xmax, ymax, 'rectangle', difficult))  # 将单个真实框加入单张图片真实框列表
    image = IMAGE(image_name, image_name_new, image_path, int(
        height), int(width), int(channels), true_box_dict_list)

    # 将单张图对象添加进全数据集数据列表中
    modify_true_box_list(image, dataset['modify_class_dict'])
    if dataset['class_pixel_distance_dict'] is not None:
        class_pixel_limit(dataset, image.true_box_list)
    if 0 == len(image.true_box_list):
        print('{} has not true box, delete!'.format(image.image_name_new))
        os.remove(image.image_path)
        process_output['no_true_box_count'] += 1
        process_output['fail_count'] += 1
        return
    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        dataset['file_prefix'] + source_annotations_name)
    if TEMP_OUTPUT(temp_annotation_output_path, image):
        process_output['temp_file_name_list'].append(image.file_name_new)
        process_output['success_count'] += 1
    else:
        process_output['fail_count'] += 1
        return

    return
