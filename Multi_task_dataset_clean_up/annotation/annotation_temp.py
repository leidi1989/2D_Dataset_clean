'''
Description:
Version:
Author: Leidi
Date: 2021-08-09 00:59:33
LastEditors: Leidi
LastEditTime: 2021-10-22 16:44:59
'''
import os
import cv2
import json

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
        image_size = cv2.imread(image_path).shape
        height = image_size[0]
        width = image_size[1]
        channels = image_size[2]
        true_segmentation_list = []
        true_box_list = []
        for obj in data['frames'][0]['objects']:
            if 'box2d' in obj:
                cls = str(obj['category'])
                cls = cls.replace(' ', '').lower()
                if cls not in dataset['detect_class_list_new']:
                    continue
                true_box_list.append(TRUE_BOX(cls,
                                              obj['box2d']['x1'],
                                              obj['box2d']['y1'],
                                              obj['box2d']['x2'],
                                              obj['box2d']['y2']))  # 将单个真实框加入单张图片真实框列表
            if 'poly2d' in obj:
                cls = str(obj['category'])
                cls = cls.replace(' ', '').lower()
                if cls not in dataset['segment_class_list_new']:
                    continue
                segment = []
                for seg in obj['poly2d']:
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
    annotation = {'name': image.file_name_new,
                  'frames': [{'timestamp': 10000,
                              'objects': []}],
                  'attributes': {'weather': 'undefined',
                                 'scene': 'city street',
                                 'timeofday': 'daytime'
                                 }
                  }
    # 目标识别真实框
    for i, n in enumerate(image.true_box_list):
        box = {'category': n.clss,
               'id': i,
               'attributes': {'occluded': False if 0 == n.difficult else str(n.difficult),
                              'truncated': False if 0 == n.occlusion else str(n.occlusion),
                              'trafficLightColor': "none"
                              },
               'box2d': {'x1': int(n.xmin),
                         'y1': int(n.ymin),
                         'x2': int(n.xmax),
                         'y2': int(n.ymax),
                         }
               }
        annotation['frames'][0]['objects'].append(box)
    # 语义分割真实框
    m = len(image.true_box_list)
    for i, n in enumerate(image.true_segmentation_list):
        segmentation = {'category': n.clss,
                        'id': i + m,
                        'attributes': {},
                        'poly2d': n.segmentation
                        }
        annotation['frames'][0]['objects'].append(segmentation)
    # 输出json文件
    json.dump(annotation, open(annotation_output_path, 'w'))

    return True
