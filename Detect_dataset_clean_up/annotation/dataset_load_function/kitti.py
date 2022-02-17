'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2022-02-17 15:44:29
'''
import os
import cv2

from utils.utils import *
from base.image_base import *
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_box_list


def load_annotation(dataset: dict, source_annotations_name: str, process_output) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        source_annotations_name (str): [源标签路径]
        process_output (dict): [进程通信字典]
    """

    source_annotation_path = os.path.join(
        dataset['source_annotations_folder'], source_annotations_name)
    with open(source_annotation_path, 'r') as f:
        true_box_dict_list = []
        for one_bbox in f.read().splitlines():
            one_bbox = one_bbox.split(' ')
            image_name = (source_annotation_path.split(
                os.sep)[-1]).replace('.txt', '.jpg')
            image_name_new = dataset['file_prefix'] + image_name
            image_path = os.path.join(
                dataset['temp_images_folder'], image_name_new)
            img = cv2.imread(image_path)
            if img is None:
                print('Can not load: {}'.format(image_name_new))
                continue
            height, width, channels = img.shape     # 读取每张图片的shape
            cls = str(one_bbox[0])
            cls = cls.strip(' ').lower()
            if cls == 'dontcare' or cls == 'misc':
                continue
            if cls not in dataset['source_class_list']:
                continue
            xmin = min(
                max(min(float(one_bbox[4]), float(one_bbox[6])), 0.), float(width))
            ymin = min(
                max(min(float(one_bbox[5]), float(one_bbox[7])), 0.), float(height))
            xmax = max(
                min(max(float(one_bbox[6]), float(one_bbox[4])), float(width)), 0.)
            ymax = max(
                min(max(float(one_bbox[7]), float(one_bbox[5])), float(height)), 0.)
            true_box_dict_list.append(TRUE_BOX(
                cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表

    image = IMAGE(image_name, image_name_new, image_path, int(
        height), int(width), int(channels), true_box_dict_list)

    # 输出读取的source annotation至temp annotation
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
