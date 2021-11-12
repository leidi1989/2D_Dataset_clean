'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:43:21
LastEditors: Leidi
LastEditTime: 2021-10-24 14:16:20
'''
import os
import cv2
import json
import random
import xml.etree.ElementTree as ET

from base.image_base import *
from utils.utils import *
from utils.convertion_function import coco_voc, revers_yolo


def YOLO_CHECK(dataset):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    check_images_list = []
    dataset['check_file_name_list'] = annotations_path_list(
        dataset['total_file_name_path'], dataset['target_annotation_check_count'])
    for n in dataset['check_file_name_list']:
        target_annotation_path = os.path.join(
            dataset['target_annotations_folder'],
            n + '.' + dataset['target_annotation_form'])
        with open(target_annotation_path, 'r') as f:
            image_name = n + '.' + dataset['target_image_form']
            image_path = os.path.join(
                dataset['temp_images_folder'], image_name)
            img = cv2.imread(image_path)
            size = img.shape
            width = int(size[1])
            height = int(size[0])
            channels = int(size[2])
            truebox_dict_list = []
            for one_bbox in f.read().splitlines():
                true_box = one_bbox.split(' ')[1:]
                cls = dataset['class_list_new'][int(one_bbox.split(' ')[0])]
                cls = cls.strip(' ').lower()
                if cls not in dataset['class_list_new']:
                    continue
                true_box = revers_yolo(size, true_box)
                xmin = min(
                    max(min(float(true_box[0]), float(true_box[1])), 0.), float(width))
                ymin = min(
                    max(min(float(true_box[2]), float(true_box[3])), 0.), float(height))
                xmax = max(
                    min(max(float(true_box[1]), float(true_box[0])), float(width)), 0.)
                ymax = max(
                    min(max(float(true_box[3]), float(true_box[2])), float(height)), 0.)
                truebox_dict_list.append(TRUE_BOX(
                    cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
            image = IMAGE(image_name, image_name, image_path, int(
                height), int(width), int(channels), truebox_dict_list)
            check_images_list.append(image)

    return check_images_list


def PASCAL_VOC_CHECK(dataset: dict) -> list:
    """[PASCAL VOC数据集annotation读取]

    Args:
        source_image_folder (str): [源数据集图片文件夹路径]
        class_names_list (list): [源数据集类别列表]
        image_name_new (str): [更名后的图片名称]
        source_annotation_path (str): [源annotation路径地址]

    Returns:
        IMAGE: [IMAGE实例]
    """
    check_images_list = []
    dataset['check_file_name_list'] = annotations_path_list(
        dataset['total_file_name_path'], dataset['target_annotation_check_count'])
    for n in dataset['check_file_name_list']:
        target_annotation_path = os.path.join(
            dataset['target_annotations_folder'],
            n + '.' + dataset['target_annotation_form'])
        tree = ET.parse(target_annotation_path)
        root = tree.getroot()
        image_name = str(root.find('filename').text)
        image_path = os.path.join(
            dataset['temp_images_folder'], image_name)
        image_size = cv2.imread(image_path).shape
        height = image_size[0]
        width = image_size[1]
        channels = image_size[2]
        truebox_dict_list = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = str(obj.find('name').text)
            cls = cls.replace(' ', '').lower()
            if cls not in dataset['class_list_new']:
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
            truebox_dict_list.append(TRUE_BOX(
                cls, xmin, ymin, xmax, ymax, 'rectangle', difficult))  # 将单个真实框加入单张图片真实框列表
        image = IMAGE(image_name, image_name, image_path, int(
            height), int(width), int(channels), truebox_dict_list)
        check_images_list.append(image)

    return check_images_list


def COCO_2017_CHECK(dataset) -> None:
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    check_images_list = []
    key = 'file_name'
    dataset['check_file_name_list'] = os.listdir(
        dataset['target_annotations_folder'])  # 读取target_annotations_folder文件夹下的全部文件名
    images_data_list = []
    for target_annotation in dataset['check_file_name_list']:
        if target_annotation != 'train.json':
            continue
        target_annotation_path = os.path.join(
            dataset['target_annotations_folder'], target_annotation)
        with open(target_annotation_path, 'r') as f:
            data = json.loads(f.read())
        max_image_id = 0
        for one_image in data['images']:    # 获取数据集image中最大id数
            max_image_id = max(max_image_id, one_image['id'])
        for _ in range(max_image_id):   # 创建全图片列表
            images_data_list.append(None)
        name_dict = {}
        for one_name in data['categories']:
            name_dict['%s' % one_name['id']] = one_name['name']
        print('Start to load each annotation data file:')
        for d in tqdm(data['images']):   # 获取data字典中images内的图片信息，file_name、height、width
            img_id = d['id']
            img_name = d[key]
            img_name_new = img_name
            img_path = os.path.join(
                dataset['temp_images_folder'], d[key])
            img = cv2.imread(img_path)
            _, _, channels = img.shape
            width = d['width']
            height = d['height']
            # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
            # 并将初始化后的对象存入total_images_data_list
            one_image = IMAGE(
                img_name, img_name_new, img_path, height, width, channels, [])
            images_data_list[img_id - 1] = one_image
        for one_annotation in tqdm(data['annotations']):
            ann_image_id = one_annotation['image_id']   # 获取此bbox图片id
            cls = name_dict[str(one_annotation['category_id'])]     # 获取bbox类别
            cls = cls.replace(' ', '').lower()
            if cls not in dataset['class_list_new']:
                continue
            # 将coco格式的bbox坐标转换为voc格式的bbox坐标，即xmin, xmax, ymin, ymax
            one_bbox_list = coco_voc(one_annotation['bbox'])
            # 为annotation对应的图片添加真实框信息
            one_bbox = TRUE_BOX(cls,
                                min(max(float(one_bbox_list[0]), 0.), float(
                                    width)),
                                max(min(
                                    float(one_bbox_list[2]), float(height)), 0.),
                                min(max(float(one_bbox_list[1]), 0.), float(
                                    width)),
                                max(min(float(one_bbox_list[3]), float(height)), 0.))
            images_data_list[ann_image_id -
                             1].true_box_list_updata(one_bbox)

    random.shuffle(images_data_list)
    check_images_list = images_data_list[0:10]

    return check_images_list


annotation_check_function_dict = {'pascal_voc': PASCAL_VOC_CHECK,
                                  'coco2017': COCO_2017_CHECK,
                                  'yolo': YOLO_CHECK,
                                  }


def annotation_check_function(dataset_stype, *args):
    """根据输出类别挑选数据集标签检测函数"""

    return annotation_check_function_dict.get(dataset_stype)(*args)
