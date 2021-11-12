'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2021-11-08 17:27:59
'''
import os
import cv2
import json
import operator
import xml.etree.ElementTree as ET

from utils.utils import *
from base.image_base import *
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_segmentation_list


def load_annotation(dataset: dict, source_annotation_name: str, process_output: dict) -> None:
    """[单进程读取标签]

    Args:
        dataset (dict): [数据集信息字典]
        source_annotation_path (str): [源标签路径]
        process_output (dict): [进程通信字典]
    """

    source_annotations_path = os.path.join(
        dataset['source_annotations_folder'], source_annotation_name)
    tree = ET.parse(source_annotations_path)
    root = tree.getroot()
    for annotation in root:
        if annotation.tag != 'image':
            continue
        image_name = str(annotation.attrib['name']).replace(
            '.' + dataset['source_image_form'], '.' + dataset['target_image_form'])
        image_name_new = dataset['file_prefix'] + image_name
        image_path = os.path.join(
            dataset['temp_images_folder'], image_name_new)
        img = cv2.imread(image_path)
        if img is None:
            print('Can not load: {}'.format(image_name_new))
            return
        width = int(annotation.attrib['width'])
        height = int(annotation.attrib['height'])
        channels = img.shape[-1]
        true_segmentation_list = []
        for obj in annotation:
            cls = str(obj.attrib['label'])
            cls = cls.replace(' ', '').lower()
            if cls not in dataset['source_class_list']:
                continue
            segment = []
            for seg in obj.attrib['points'].split(';'):
                x, y = seg.split(',')
                x = float(x)
                y = float(y)
                segment.append(list(map(int, [x, y])))
            true_segmentation_list.append(TRUE_SEGMENTATION(
                cls, segment))  # 将单个真实框加入单张图片真实框列表
        image = IMAGE(image_name, image_name_new, image_path, int(
            height), int(width), int(channels), [], true_segmentation_list)
        
        modify_true_segmentation_list(image, dataset['modify_class_dict'])
        if dataset['class_pixel_distance_dict'] is not None:
            class_segmentation_pixel_limit(dataset, image.true_box_list)
        if 0 == len(image.true_segmentation_list):
            print('{} has not true box, delete!'.format(image.image_name_new))
            os.remove(image.image_path)
            process_output['no_segmentation'] += 1
            process_output['fail_count'] += 1
            return
        temp_annotation_output_path = os.path.join(
            dataset['temp_annotations_folder'],
            image.file_name_new + '.' + dataset['temp_annotation_form'])
        if TEMP_OUTPUT(temp_annotation_output_path, image):
            process_output['temp_file_name_list'].append(image.file_name_new)
            process_output['success_count'] += 1
        else:
            process_output['fail_count'] += 1
            return

    return
