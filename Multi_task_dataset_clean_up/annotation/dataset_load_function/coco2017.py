'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2021-10-23 00:46:01
'''
import os
from PIL import Image

from utils.utils import *
from base.image_base import *
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_box_list, modify_true_segmentation_list


def load_image_base_information(dataset: dict, one_image_base_information: dict, each_annotation_images_data_dict: dict) -> None:
    """[读取标签获取图片基础信息，并添加至each_annotation_images_data_dict]

    Args:
        dataset (dict): [数据集信息字典]
        one_image_base_information (dict): [单个数据字典信息]
        each_annotation_images_data_dict进程间通信字典 (dict): [each_annotation_images_data_dict进程间通信字典]
    """

    image_id = one_image_base_information['id']
    image_name = one_image_base_information['file_name'].split(
        '.')[0] + '.' + dataset['temp_image_form']
    image_name_new = dataset['file_prefix'] + image_name
    image_path = os.path.join(dataset['temp_images_folder'], image_name_new)
    img = Image.open(image_path)
    height, width = img.height, img.width
    channels = 3
    # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
    # 并将初始化后的对象存入total_images_data_list
    image = IMAGE(image_name, image_name_new,
                  image_path, height, width, channels, [], [])
    each_annotation_images_data_dict.update({image_id: image})

    return


def load_image_annotation(dataset: dict, one_annotation: dict, class_dict: dict, each_annotation_images_data_dict: dict) -> list:
    """[读取单个标签详细信息，并添加至each_annotation_images_data_dict]

    Args:
        dataset (dict): [数据集信息字典]
        one_annotation (dict): [单个数据字典信息]
        class_dict (dict): [description]
        process_output (dict): [each_annotation_images_data_dict进程间通信字典]

    Returns:
        list: [ann_image_id, true_box_list, true_segmentation_list]
    """    

    ann_image_id = one_annotation['image_id']   # 获取此bbox图片id
    cls = class_dict[str(one_annotation['category_id'])]     # 获取bbox类别
    cls = cls.replace(' ', '').lower()
    if cls not in dataset['source_segment_class_list']:
        return
    image = each_annotation_images_data_dict[ann_image_id]

    # 获取真实框信息
    true_box_list = []
    if 'bbox' in one_annotation and len(one_annotation['bbox']):
        box = [one_annotation['bbox'][0],
               one_annotation['bbox'][1],
               one_annotation['bbox'][0] + one_annotation['bbox'][2],
               one_annotation['bbox'][1] + one_annotation['bbox'][3]]
        xmin = max(min(int(box[0]), int(box[2]),
                       int(image.width)), 0.)
        ymin = max(min(int(box[1]), int(box[3]),
                       int(image.height)), 0.)
        xmax = min(max(int(box[2]), int(box[0]), 0.),
                   int(image.width))
        ymax = min(max(int(box[3]), int(box[1]), 0.),
                   int(image.height))
        true_box_list.append(
            TRUE_BOX(cls, xmin, ymin, xmax, ymax))

    # 获取真实语义分割信息
    true_segmentation_list = []
    for one_seg in one_annotation['segmentation']:
        segment = []
        point = []
        for i, x in enumerate(one_seg):
            if 0 == i % 2:
                point.append(x)
            else:
                point.append(x)
                point = list(map(int, point))
                segment.append(point)
                if 2 != len(point):
                    print('Segmentation label wrong: ',
                          each_annotation_images_data_dict[ann_image_id].image_name_new)
                    continue
                point = []
        if 0 == one_annotation['iscrowd']:
            true_segmentation_list.append(TRUE_SEGMENTATION(
                cls, segment, one_annotation['area']))
        else:
            true_segmentation_list.append(TRUE_SEGMENTATION(
                cls, segment, one_annotation['area'], 1))

    return ann_image_id, true_box_list, true_segmentation_list


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
    modify_true_box_list(image, dataset['detect_modify_class_dict'])
    modify_true_segmentation_list(image, dataset['segment_modify_class_dict'])
    if dataset['class_pixel_distance_dict'] is not None:
        class_box_pixel_limit(dataset, image.true_box_list)
        class_segmentation_pixel_limit(dataset, image.true_segmentation_list)
    if 0 == len(image.true_segmentation_list) and 0 == len(image.true_box_list):
        print('{} no true box and segmentation, has been delete.'.format(
            image.image_name_new))
        os.remove(image.image_path)
        process_output['no_detect_segmentation'] += 1
        process_output['fail_count'] += 1
        return
    if TEMP_OUTPUT(temp_annotation_output_path, image):
        process_output['temp_file_name_list'].append(image.file_name_new)
        process_output['success_count'] += 1
    else:
        process_output['fail_count'] += 1

    return
