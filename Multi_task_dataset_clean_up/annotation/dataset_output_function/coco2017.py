'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-12-22 15:33:17
'''
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


def get_annotation(dataset: dict, n: int, temp_annotation_path: str, process_annotation_count: dict) -> None:
    """[获取暂存标注信息]

    Args:
        dataset (dict): [数据集信息字典]
        n (int): [图片id]
        temp_annotation_path (str): [暂存标签路径]
        process_annotation_count (dict): [annotation_count进程间通信字典]
    """

    image = TEMP_LOAD(dataset, temp_annotation_path)
    if image == None:
        return

    # 获取图片分割信息
    one_image_annotations_list = []
    if len(image.true_segmentation_list):
        for true_segmentation in image.true_segmentation_list:
            segmentation = np.asarray(
                true_segmentation.segmentation).flatten().tolist()
            one_image_annotations_list.append({'segmentation': [segmentation],
                                               'bbox': [],
                                               'area': 0,
                                               'iscrowd': true_segmentation.iscrowd,
                                               'image_id': n,
                                               'category_id': (dataset['detect_class_list_new'] + dataset['segment_class_list_new']).index(true_segmentation.clss),
                                               'id': process_annotation_count['annotation_count']})
            process_annotation_count['annotation_count'] += 1

    if len(image.true_box_list):
        for true_box in image.true_box_list:
            bbox = [true_box.xmin,
                    true_box.ymin,
                    true_box.xmax-true_box.xmin,
                    true_box.ymax-true_box.ymin,
                    ]
            one_image_annotations_list.append({'segmentation': [],
                                               'bbox': bbox,
                                               'area': 0,
                                               'iscrowd': 0,
                                               'image_id': n,
                                               'category_id': (dataset['detect_class_list_new'] + dataset['segment_class_list_new']).index(true_segmentation.clss),
                                               'id': process_annotation_count['annotation_count']})
            process_annotation_count['annotation_count'] += 1

    return one_image_annotations_list
