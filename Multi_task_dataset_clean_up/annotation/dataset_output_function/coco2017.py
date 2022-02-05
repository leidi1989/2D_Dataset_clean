'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-12-28 16:19:03
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
    if len(image.true_segmentation_list):
        for true_segmentation in image.true_segmentation_list:
            bbox = true_segmentation_to_true_box(true_segmentation)
            segmentation = np.asarray(
                true_segmentation.segmentation).flatten().tolist()
            area = int(cv2.contourArea(
                np.array(true_segmentation.segmentation)))
            one_image_annotations_list.append({'segmentation': [segmentation],
                                               'bbox': bbox,
                                               'area': area,
                                               'iscrowd': true_segmentation.iscrowd,
                                               'image_id': n,
                                               'category_id': (dataset['detect_class_list_new'] + dataset['segment_class_list_new']).index(true_segmentation.clss),
                                               'id': 0})
    # 图片真实框信息
    if len(image.true_box_list):
        for true_box in image.true_box_list:
            bbox = [int(true_box.xmin),
                    int(true_box.ymin),
                    int(true_box.xmax-true_box.xmin),
                    int(true_box.ymax-true_box.ymin),
                    ]
            segmentation = [str(true_box.xmin), str(
                true_box.ymin), str(true_box.xmax), str(true_box.ymax)]
            area = int(true_box.xmax-true_box.xmin) * \
                int(true_box.ymax-true_box.ymin)
            one_image_annotations_list.append({'segmentation': [segmentation],
                                               'bbox': bbox,
                                               'area': area,
                                               'iscrowd': 0,
                                               'image_id': n,
                                               'category_id': (dataset['detect_class_list_new'] + dataset['segment_class_list_new']).index(true_segmentation.clss),
                                               'id': 0})

    return one_image_annotations_list
