'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2021-12-31 15:28:37
'''
import os
from PIL import Image

from utils.utils import *
from base.image_base import *
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_segmentation_list
from utils.convertion_function import true_segmentation_to_true_box


def load_annotation(dataset: dict, source_annotations_name: str, process_output) -> None:
    """[读取yunce_segment_coco_one_image标签]

    Args:
        dataset (dict): [数据集信息字典]
        source_annotations_path (str): [源标签路径]
        process_output ([dict]): [进程通信字典]
    """

    source_annotation_path = os.path.join(
        dataset['source_annotations_folder'], source_annotations_name)
    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        dataset['file_prefix'] + source_annotations_name)
    with open(source_annotation_path, 'r') as f:
        data = json.loads(f.read())

        class_dict = {}
        for n in data['categories']:
            class_dict['%s' % n['id']] = n['name']

        image_name = os.path.splitext(data['images'][0]['file_name'])[
            0] + '.' + dataset['temp_image_form']
        image_name_new = dataset['file_prefix'] + image_name
        image_path = os.path.join(
            dataset['temp_images_folder'], image_name_new)
        img = Image.open(image_path)
        height = img.height
        width = img.width
        channels = len(img.split())
        true_segmentation_list = []
        for obj in data['annotations']:
            cls = class_dict[str(obj['category_id'])]     # 获取bbox类别
            cls = cls.replace(' ', '').lower()
            if cls == 'static_object.concave.firehydrant':
                cls = 'static_object.concave.fire_hydrant'
            if cls == 'static_object.concave.firehydrant_infer':
                cls = 'static_object.concave.fire_hydrant_infer'
            if cls not in dataset['source_class_list']:
                continue
            segment = []
            c = 0
            for n in obj['segmentation']:
                if 0 == c:
                    segment.append([])
                    segment[-1].append(n)
                    c += 1
                else:
                    segment[-1].append(n)
                    c = 0
            true_segmentation = TRUE_SEGMENTATION(
                cls, segment)
            true_segmentation_list.append(true_segmentation)
        image = IMAGE(image_name, image_name_new, image_path, int(
            height), int(width), int(channels), [], true_segmentation_list)
        f.close()
    # 输出读取的source annotation至temp annotation
    if image == None:
        return
    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        image.file_name_new + '.' + dataset['temp_annotation_form'])
    modify_true_segmentation_list(image, dataset['modify_class_dict'])
    if dataset['class_pixel_distance_dict'] is not None:
        class_segmentation_pixel_limit(dataset, image.true_segmentation_list)
    if 0 == len(image.true_segmentation_list) and 0 == len(image.true_box_list):
        print('{} has not true segmentation and box, delete!'.format(
            image.image_name_new))
        os.remove(image.image_path)
        process_output['no_segmentation'] += 1
        process_output['fail_count'] += 1
        return
    if TEMP_OUTPUT(temp_annotation_output_path, image):
        process_output['temp_file_name_list'].append(image.file_name_new)
        process_output['success_count'] += 1
    else:
        process_output['fail_count'] += 1
        return

    return
