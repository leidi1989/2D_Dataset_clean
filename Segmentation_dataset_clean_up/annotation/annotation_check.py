'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:43:21
LastEditors: Leidi
LastEditTime: 2021-11-22 09:53:50
'''
import os
import cv2
import json

from utils.utils import *
from base.image_base import *


def cityscapes(dataset: dict) -> list:
    """[读取cityscapes数据集图片类检测列表]

    Args:
        dataset (dict): [数据集信息字典]

    Returns:
        list: [数据集图片类检测列表]
    """

    check_images_list = []
    dataset['total_file_name_path'] = os.path.join(
        dataset['temp_informations_folder'], 'total_file_name.txt')
    dataset['check_file_name_list'] = annotations_path_list(
        dataset['total_file_name_path'], dataset['target_annotation_check_count'])
    print('Start load target annotations:')
    for n in tqdm(dataset['check_file_name_list']):
        target_annotation_path = os.path.join(
            dataset['target_annotations_folder'],
            n + '.' + dataset['target_annotation_form'])
        try:    
            with open(target_annotation_path, 'r') as f:
                data = json.loads(f.read())
                image_name = n + '.' + dataset['target_image_form']
                image_path = os.path.join(
                    dataset['temp_images_folder'], image_name)
                image_size = cv2.imread(image_path).shape
                height = image_size[0]
                width = image_size[1]
                channels = image_size[2]
                true_segmentation_list = []
                for obj in data['objects']:
                    cls = str(obj['label'])
                    cls = cls.replace(' ', '').lower()
                    if cls not in dataset['class_list_new']:
                        continue
                    true_segmentation_list.append(TRUE_SEGMENTATION(
                        cls, obj['polygon']))  # 将单个真实框加入单张图片真实框列表
                image = IMAGE(image_name, image_name, image_path, int(
                    height), int(width), int(channels), [], true_segmentation_list)
                check_images_list.append(image)
        except:
            print('\nLoad erro:', target_annotation_path)

    return check_images_list


def cityscapes_val(dataset: dict) -> list:
    """[读取cityscapes数据集图片类检测列表]

    Args:
        dataset (dict): [数据集信息字典]

    Returns:
        list: [数据集图片类检测列表]
    """

    return []


# def COCO_2017_CHECK(dataset: dict) -> list:
#     """[读取COCO2017数据集图片类检测列表]

#     Args:
#         dataset (dict): [数据集信息字典]

#     Returns:
#         list: [数据集图片类检测列表]
#     """

#     check_images_list = []
#     key = 'file_name'
#     dataset['check_file_name_list'] = os.listdir(
#         dataset['target_annotations_folder'])  # 读取target_annotations_folder文件夹下的全部文件名
#     images_data_list = []
#     for target_annotation in dataset['check_file_name_list']:
#         if target_annotation != 'train.json':
#             continue
#         target_annotation_path = os.path.join(
#             dataset['target_annotations_folder'], target_annotation)
#         with open(target_annotation_path, 'r') as f:
#             data = json.loads(f.read())
#         max_image_id = 0
#         for one_image in data['images']:    # 获取数据集image中最大id数
#             max_image_id = max(max_image_id, one_image['id'])
#         for _ in range(max_image_id):   # 创建全图片列表
#             images_data_list.append(None)
#         name_dict = {}
#         for one_name in data['categories']:
#             name_dict['%s' % one_name['id']] = one_name['name']
#         print('Start to load each annotation data file:')
#         for d in tqdm(data['images']):   # 获取data字典中images内的图片信息，file_name、height、width
#             img_id = d['id']
#             img_name = d[key]
#             img_name_new = img_name
#             img_path = os.path.join(
#                 dataset['temp_images_folder'], d[key])
#             img = cv2.imread(img_path)
#             _, _, channels = img.shape
#             width = d['width']
#             height = d['height']
#             # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
#             # 并将初始化后的对象存入total_images_data_list
#             one_image = IMAGE(
#                 img_name, img_name_new, img_path, height, width, channels, [])
#             images_data_list[img_id - 1] = one_image
#         for one_annotation in tqdm(data['annotations']):
#             ann_image_id = one_annotation['image_id']   # 获取此bbox图片id
#             cls = name_dict[str(one_annotation['category_id'])]     # 获取bbox类别
#             cls = cls.replace(' ', '').lower()
#             if cls not in dataset['class_list_new']:
#                 continue
#             # 将coco格式的bbox坐标转换为voc格式的bbox坐标，即xmin, xmax, ymin, ymax
#             one_bbox_list = coco_voc(one_annotation['bbox'])
#             # 为annotation对应的图片添加真实框信息
#             one_bbox = TRUE_BOX(cls,
#                                 min(max(float(one_bbox_list[0]), 0.), float(
#                                     width)),
#                                 max(min(
#                                     float(one_bbox_list[2]), float(height)), 0.),
#                                 min(max(float(one_bbox_list[1]), 0.), float(
#                                     width)),
#                                 max(min(float(one_bbox_list[3]), float(height)), 0.))
#             images_data_list[ann_image_id -
#                              1].true_box_list_updata(one_bbox)

#     random.shuffle(images_data_list)
#     check_images_list = images_data_list[0:10]

#     return check_images_list
