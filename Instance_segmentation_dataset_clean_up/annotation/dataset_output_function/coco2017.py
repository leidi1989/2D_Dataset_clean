'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-12-31 16:25:00
'''
import cv2
import time
import numpy as np

from utils.utils import *
from base.image_base import *
from utils.convertion_function import *
from annotation.annotation_temp import TEMP_LOAD


def get_image_information(dataset: dict, coco: dict, n: int, temp_annotation_path: str) -> None:
    """[获取暂存标注信息]

    Args:
        dataset (dict): [数据集信息字典]
        coco (dict): [coco汇总字典]
        n (int): [图片id]
        temp_annotation_path (str): [暂存标签路径]
    """

    image = TEMP_LOAD(dataset, temp_annotation_path)
    if image == None:
        return
    # 图片基础信息
    image_information = {'license': random.randint(0, len(coco['licenses'])),
                         'file_name': image.image_name_new,
                         'coco_url': 'None',
                         'height': image.height,
                         'width': image.width,
                         'date_captured': time.strftime('%Y/%m/%d %H:%M:%S', time.localtime()),
                         'flickr_url': 'None',
                         'id': n
                         }

    return image_information


def get_annotation(dataset: dict, n: int, temp_annotation_path: str) -> None:
    """[获取暂存标注信息]

    Args:
        dataset (dict): [数据集信息字典]
        n (int): [图片id]
        temp_annotation_path (str): [暂存标签路径]
    """

    image = TEMP_LOAD(dataset, temp_annotation_path)
    if image == None:
        return
    # 获取图片分割信息
    one_image_annotations_list = []
    for true_segmentation in image.true_segmentation_list:
        bbox = temp_box_to_coco_box(true_segmentation.segmentation_bounding_box)
        segmentation = np.asarray(
            true_segmentation.segmentation).flatten().tolist()
        one_image_annotations_list.append({'segmentation': [segmentation],
                                           'bbox': bbox,
                                           'area': true_segmentation.area,
                                           'iscrowd': true_segmentation.iscrowd,
                                           'image_id': n,
                                           'category_id': dataset['class_list_new'].index(true_segmentation.clss),
                                           'id': 0})

    return one_image_annotations_list
