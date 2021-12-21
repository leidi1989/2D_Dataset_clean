'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-12-21 19:19:37
'''
import os
import json
import time
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

from utils.utils import *
from base.image_base import *
from utils.convertion_function import *
from annotation.annotation_temp import TEMP_LOAD


def annotation_output(dataset: dict, coco: dict, n: int, temp_annotation_path: str,
                      process_images: list, process_annotations: list, process_annotation_count: dict) -> None:

    image = TEMP_LOAD(dataset, temp_annotation_path)
    if image == None:
        return
    # 图片基础信息
    process_images.append({'license': random.randint(0, len(coco['licenses'])),
                           'file_name': image.image_name_new,
                           'coco_url': 'None',
                           'height': image.height,
                           'width': image.width,
                           'date_captured': time.strftime('%Y/%m/%d %H:%M:%S', time.localtime()),
                           'flickr_url': 'None',
                           'id': n
                           })
    # 分割信息
    for true_segmentation in image.true_segmentation_list:
        segmentation = np.asarray(
            true_segmentation.segmentation).flatten().tolist()
        process_annotations.append({'segmentation': [segmentation],
                                    'area': 0,
                                    'iscrowd': true_segmentation.iscrowd,
                                    'image_id': n,
                                    'category_id': dataset['class_list_new'].index(true_segmentation.clss),
                                    'id': process_annotation_count['annotation_count']})
        process_annotation_count['annotation_count'] += 1

    return
