'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:59:27
LastEditors: Leidi
LastEditTime: 2021-10-28 14:43:43
'''
import os
import cv2

from utils.utils import *
from base.image_base import *
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_box_list


def load_annotation(dataset: dict, image_id: int, out_image: dict, process_output) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        image_id (int): [图片id]
        out_image (dict): [图片标签信息]
        process_output (dict): [进程通信字典]
    """

    if 0 == len(out_image['objects']):
        return
    image_name = str(image_id) + '.' + dataset['target_image_form']
    image_name_new = dataset['file_prefix'] + image_name
    image_path = os.path.join(
        dataset['temp_images_folder'], image_name_new)
    img = cv2.imread(image_path)
    if img is None:
        print('Can not load: {}'.format(image_name_new))
        return
    height, width, channels = img.shape
    true_box_dict_list = []
    for m in out_image['objects']:
        cls = str(m['category'])
        cls = cls.replace(' ', '').lower()
        if cls not in dataset['source_class_list']:
            continue
        true_box = m['bbox']
        box = (int(true_box['xmin']),
               int(true_box['xmax']),
               int(true_box['ymin']),
               int(true_box['ymax']))
        xmin = max(min(int(box[0]), int(box[1]), int(width)), 0.)
        ymin = max(min(int(box[2]), int(box[3]), int(height)), 0.)
        xmax = min(max(int(box[1]), int(box[0]), 0.), int(width))
        ymax = min(max(int(box[3]), int(box[2]), 0.), int(height))
        # 将单个真实框加入单张图片真实框列表
        true_box_dict_list.append(TRUE_BOX(
            cls, xmin, ymin, xmax, ymax))

    image = IMAGE(
        image_name, image_name_new, image_path, height, width, channels, true_box_dict_list)

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
