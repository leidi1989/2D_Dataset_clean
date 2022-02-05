'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-10-28 14:42:46
'''
import os
import cv2

from base.image_base import *
from utils.utils import class_pixel_limit
from utils.convertion_function import revers_yolo
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_box_list


def load_annotation(dataset: dict, source_annotations_name: str, process_output) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        source_annotations_name (str): [源标签名称]
        process_output ([dict]): [进程通信字典]
    """

    source_annotations_path = os.path.join(
        dataset['source_annotations_folder'], source_annotations_name)
    with open(source_annotations_path, 'r') as f:
        true_box_dict_list = []
        image_name = (source_annotations_path.split(
            '/')[-1]).replace('.txt', '.jpg')
        image_name_new = dataset['file_prefix'] + (source_annotations_path.split(
            '/')[-1]).replace('.txt', '.jpg')
        image_path = os.path.join(
            dataset['temp_images_folder'], image_name_new)
        img = cv2.imread(image_path)
        if img is None:
            print('Can not load: {}'.format(image_name_new))
            return
        height, width, channels = img.shape
        for one_bbox in f.read().splitlines():
            bbox = one_bbox.split(' ')[1:]
            cls = dataset['source_class_list'][int(one_bbox.split(' ')[0])]
            cls = cls.strip(' ').lower()
            if cls not in dataset['source_class_list']:
                continue
            if cls == 'dontcare' or cls == 'misc':
                continue
            bbox = revers_yolo(img.shape, bbox)
            xmin = min(
                max(min(float(bbox[0]), float(bbox[1])), 0.), float(width))
            ymin = min(
                max(min(float(bbox[2]), float(bbox[3])), 0.), float(height))
            xmax = max(
                min(max(float(bbox[1]), float(bbox[0])), float(width)), 0.)
            ymax = max(
                min(max(float(bbox[3]), float(bbox[2])), float(height)), 0.)
            true_box_dict_list.append(TRUE_BOX(
                cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
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
        image.file_name_new + '.' + dataset['temp_annotation_form'])
    if TEMP_OUTPUT(temp_annotation_output_path, image):
        process_output['temp_file_name_list'].append(image.file_name_new)
        process_output['success_count'] += 1
    else:
        process_output['fail_count'] += 1
        return

    return
