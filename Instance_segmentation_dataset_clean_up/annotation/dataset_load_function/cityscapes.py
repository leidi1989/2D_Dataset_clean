'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:59:27
LastEditors: Leidi
LastEditTime: 2021-10-27 15:33:33
'''
import os
import cv2
import json

from utils.utils import *
from base.image_base import *
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_segmentation_list


def load_annotation(dataset: dict, source_annotations_name: str, process_output) -> None:
    """[读取cityscapes标签]

    Args:
        dataset (dict): [数据集信息字典]
        source_annotations_path (str): [源标签路径]
        process_output ([dict]): [进程通信字典]
    """
    
    source_annotation_path = os.path.join(
        dataset['source_annotations_folder'], source_annotations_name)
    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        dataset['file_prefix'] + source_annotations_name)
    with open(source_annotation_path, 'r') as f:
        data = json.loads(f.read())
        image_name = source_annotation_path.split(
            os.sep)[-1].replace('_gtFine_polygons.json', '_leftImg8bit.png')
        image_name_new = dataset['file_prefix'] + image_name
        image_path = os.path.join(
            dataset['temp_images_folder'], image_name_new)
        image_size = cv2.imread(image_path).shape
        height = image_size[0]
        width = image_size[1]
        channels = image_size[2]
        true_segmentation_list = []
        for obj in data['objects']:
            cls = str(obj['label'])
            cls = cls.replace(' ', '').lower()
            if cls not in dataset['source_class_list']:
                continue
            segment = []
            for seg in obj['polygon']:
                segment.append(list(map(int, seg)))
            true_segmentation_list.append(TRUE_SEGMENTATION(
                cls, segment))  # 将单个真实框加入单张图片真实框列表
        image = IMAGE(image_name, image_name_new, image_path, int(
            height), int(width), int(channels), [], true_segmentation_list)
        f.close()
    # 输出读取的source annotation至temp annotation
    if image == None:
        return
    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        image.file_name_new + '.' + dataset['temp_annotation_form'])
    modify_true_segmentation_list(image, dataset['modify_class_dict'])
    if dataset['class_pixel_distance_dict'] is not None:
        class_segmentation_pixel_limit(dataset, image.true_segmentation_list)
    if 0 == len(image.true_segmentation_list) and 0 == len(image.true_box_list):
        print('{} has not true segmentation and box, delete!'.format(
            image.image_name_new))
        os.remove(image.image_path)
        process_output['no_segmentation'] += 1
        process_output['fail_count'] += 1
        return
    if TEMP_OUTPUT(temp_annotation_output_path, image):
        process_output['temp_file_name_list'].append(image.file_name_new)
        process_output['success_count'] += 1
    else:
        process_output['fail_count'] += 1
        return

    return
