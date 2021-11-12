'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-11-08 17:40:17
'''
import os
import cv2
import json
import shutil
import numpy as np

from annotation.annotation_temp import TEMP_LOAD


def create_annotation_file(dataset: dict, file_name_dict: dict, output_folder_path_list: list,
                           dataset_name: str, class_names_dict: dict, x: str, ) -> None:
    """[创建cityscapes格式数据集]

    Args:
        dataset (dict): [数据集信息字典]
        file_name_dict (dict): [全量数据编号字典]
        output_folder_path_list (list): [输出文件夹路径列表]
        dataset_name (str): [划分后数据集名称]
        class_names_dict (dict): [labelIds类别名对应id字典]
        x (str): [标签文件名称]
    """
    file = os.path.splitext(x.split(os.sep)[-1])[0]
    file_out = dataset['dataset_prefix'] + '_000000_' + \
        str(format(file_name_dict[file], '06d'))
    # 调整image
    image_out = file_out + '_leftImg8bit' + \
        '.' + dataset['target_image_form']
    image_path = os.path.join(
        dataset['temp_images_folder'], file + '.' + dataset['target_image_form'])
    image_output_path = os.path.join(
        output_folder_path_list[1], dataset_name, dataset['dataset_prefix'], image_out)
    # 调整annotation
    annotation_out = file_out + '_gtFine_polygons' + \
        '.' + dataset['target_annotation_form']
    annotation_path = os.path.join(
        dataset['target_annotations_folder'], file + '.' + dataset['target_annotation_form'])
    annotation_output_path = os.path.join(
        output_folder_path_list[0], dataset_name, dataset['dataset_prefix'], annotation_out)
    # 调整annotation为_gtFine_labelIds.png
    image = TEMP_LOAD(dataset, annotation_path)
    labelIds_out = file_out + '_gtFine_labelIds.png'
    labelIds_output_path = os.path.join(
        output_folder_path_list[0], dataset_name, dataset['dataset_prefix'], labelIds_out)
    # 输出
    shutil.copy(image_path, image_output_path)
    shutil.copy(annotation_path, annotation_output_path)

    zeros = np.zeros((image.height, image.width), dtype=np.uint8)
    if len(image.true_segmentation_list):
        for seg in image.true_segmentation_list:
            class_color = class_names_dict[seg.clss]
            points = np.array(seg.segmentation)
            zeros_mask = cv2.fillPoly(
                zeros, pts=[points], color=class_color)
        cv2.imwrite(labelIds_output_path, zeros_mask)
    else:
        cv2.imwrite(labelIds_output_path, zeros)

    return
