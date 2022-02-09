'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-10-21 19:28:14
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
    annotation = {'imgHeight': image.height,
                  'imgWidth': image.width,
                  'objects': []
                  }
    segmentation = {}
    for true_segmentation in image.true_segmentation_list:
        segmentation = {'label': true_segmentation.clss,
                        'polygon': true_segmentation.segmentation
                        }
        annotation['objects'].append(segmentation)
    json.dump(annotation, open(annotation_output_path, 'w'))

    return
