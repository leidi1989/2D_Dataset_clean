'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-12-22 11:24:16
'''
import os
import json

from annotation.annotation_temp import TEMP_LOAD


def annotation_output(dataset: dict, temp_annotation_path: str) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        temp_annotation_path (str): [暂存标签路径]
    """

    image = TEMP_LOAD(dataset, temp_annotation_path)
    if image == None:
        return
    annotation_output_path = os.path.join(
        dataset['target_annotations_folder'], image.file_name + '.' + dataset['target_annotation_form'])
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
                              'trafficLightColor': 'none'
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

    return
