'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-09 00:59:33
LastEditors: Leidi
LastEditTime: 2021-12-31 15:11:20
'''
import os
import cv2
import json
from PIL import Image

from base.image_base import *


def TEMP_LOAD(dataset: dict, temp_annotation_path: str) -> IMAGE:
    """[读取暂存annotation]

    Args:
        dataset (dict): [数据集信息字典]
        temp_annotation_path (str): [annotation路径]

    Returns:
        IMAGE: [输出IMAGE类变量]
    """

    with open(temp_annotation_path, 'r') as f:
        data = json.loads(f.read())
        image_name = temp_annotation_path.split(
            os.sep)[-1].replace('.json', '.' + dataset['temp_image_form'])
        image_path = os.path.join(dataset['temp_images_folder'], image_name)
        if os.path.splitext(image_path)[-1] == 'png':
            img = Image.open(image_path)
            height, width = img.height, img.width
            channels = 3
        else:
            image_size = cv2.imread(image_path).shape
            height = image_size[0]
            width = image_size[1]
            channels = image_size[2]

        true_box_list = []
        for obj_box in data['objects_box']:
            cls = str(obj_box['class'])
            cls = cls.replace(' ', '').lower()
            if cls not in dataset['class_list_new']:
                continue
            true_box_list.append(TRUE_BOX(
                cls, obj_box['xmin'], obj_box['ymin'], obj_box['xmax'], obj_box['ymax'],
                obj_box['color'], obj_box['tool'], obj_box['occluded'], obj_box['truncated']))

        true_segmentation_list = []
        for obj_segment in data['objects_segment']:
            cls = str(obj_segment['class'])
            cls = cls.replace(' ', '').lower()
            if cls not in dataset['class_list_new']:
                continue
            segment = []
            for seg in obj_segment['polygon']:
                segment.append(list(map(int, seg)))
            true_segmentation_list.append(TRUE_SEGMENTATION(
                cls, segment))  # 将单个真实框加入单张图片真实框列表
        one_image = IMAGE(image_name, image_name, image_path, int(
            height), int(width), int(channels), true_box_list, true_segmentation_list)
        f.close()

    return one_image


def TEMP_OUTPUT(annotation_output_path: str, image: IMAGE) -> bool:
    """[输出temp dataset annotation]

    Args:
        dataset (dict): [数据集信息字典]
        annotation_output_path (str): [temp dataset annotation输出路径]
        image (IMAGE): [IMAGE实例]

    Returns:
        bool: [输出是否成功]
    """

    if image == None:
        return False
    annotation = {'imgHeight': image.height,
                  'imgWidth': image.width,
                  'objects_box': [],
                  'objects_segment': []
                  }

    box = {}
    for true_box in image.true_box_list:
        box = {'class': true_box.clss,
               'xmin': int(true_box.xmin),
               'ymin': int(true_box.ymin),
               'xmax': int(true_box.xmax),
               'ymax': int(true_box.ymax),
               'color': true_box.color,
               'tool': true_box.tool,
               'occluded': False if 0 == true_box.difficult else str(true_box.difficult),
               'truncated': False if 0 == true_box.occlusion else str(true_box.occlusion),
               }
        annotation['objects_box'].append(box)

    segmentation = {}
    for true_segmentation in image.true_segmentation_list:
        segmentation = {'class': true_segmentation.clss,
                        'polygon': true_segmentation.segmentation
                        }
        annotation['objects_segment'].append(segmentation)
    json.dump(annotation, open(annotation_output_path, 'w'))

    return True
