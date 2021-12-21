'''
Description:
Version:
Author: Leidi
Date: 2021-08-11 03:28:09
LastEditors: Leidi
LastEditTime: 2021-12-21 20:15:03
'''
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import multiprocessing
from shutil import copyfile

from base.image_base import *
from utils.utils import check_output_path, err_call_back

import out.framwork_update_function as F


def cityscapes(dataset: dict) -> None:
    """[生成cityscapes组织结构数据集]

    Args:
        dataset (dict): [数据集信息字典]
    """

    # 官方数值
    # colors = [
    #     [128, 64, 128], [244, 35, 232], [70, 70, 70], [
    #         102, 102, 156], [190, 153, 153],
    #     [153, 153, 153], [250, 170, 30], [220, 220, 0], [
    #         107, 142, 35],  [152, 251, 152],
    #     [0, 130, 180],  [220, 20, 60],  [
    #         255, 0, 0],  [0, 0, 142],     [0, 0, 70],
    #     [0, 60, 100],   [0, 80, 100],   [0, 0, 230],  [119, 11, 32], [0, 0, 0]]
    # label_colours = dict(zip(range(19), colors))
    # void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    # valid_classes = [7, 8, 11, 12, 13, 17, 19, 20,
    #                  21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    # class_map = dict(zip(valid_classes, range(19)))
    # class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
    #                'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
    #                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    #                'motorcycle', 'bicycle']
    # ignore_index = 255
    # num_classes_ = 19
    # class_weights_ = np.array([0.05570516, 0.32337477, 0.08998544, 1.03602707, 1.03413147, 1.68195437,
    #                            5.58540548, 3.56563995, 0.12704978, 1., 0.46783719, 1.34551528,
    #                            5.29974114, 0.28342531, 0.9396095, 0.81551811, 0.42679146, 3.6399074,
    #                            2.78376194], dtype=float)
    # labelIds
    # class_names_dict = {'unlabeled': 0,
    #                     'egovehicle': 1,
    #                     'rectificationborder': 2,
    #                     'outofroi': 3,
    #                     'static': 4,
    #                     'dynamic': 5,
    #                     'ground': 6,
    #                     'road': 7,
    #                     'sidewalk': 8,
    #                     'parking': 9,
    #                     'railtrack': 10,
    #                     'building': 11,
    #                     'wall': 12,
    #                     'fence': 13,
    #                     'guardrail': 14,
    #                     'bridge': 15,
    #                     'tunnel': 16,
    #                     'pole': 17,
    #                     'polegroup': 18,
    #                     'trafficlight': 19,
    #                     'trafficsign': 20,
    #                     'vegetation': 21,
    #                     'terrain': 22,
    #                     'sky': 23,
    #                     'person': 24,
    #                     'rider': 25,
    #                     'car': 26,
    #                     'truck': 27,
    #                     'bus': 28,
    #                     'caravan': 29,
    #                     'trailer': 30,
    #                     'train': 31,
    #                     'motorcycle': 32,
    #                     'bicycle': 33,
    #                     'licenseplate': -1,
    #                     }
    # road lane classes
    # class_names_dict = {'unlabeled': 0,
    #                     'road': 1,
    #                     'lane': 2,
    #                     }
    class_names_dict = {}
    for x, cls in enumerate(dataset['class_list_new']):
        class_names_dict.update({cls: x})

    # 获取全量数据编号字典
    file_name_dict = {}
    print('Collect file name dict.')
    with open(dataset['temp_divide_file_list'][0], 'r') as f:
        for x, n in enumerate(f.read().splitlines()):
            file_name = os.path.splitext(n.split(os.sep)[-1])[0]
            file_name_dict[file_name] = x
        f.close()

    output_root = check_output_path(os.path.join(
        dataset['target_path'], 'cityscapes', 'data'))   # 输出数据集文件夹
    cityscapes_folder_list = ['gtFine', 'leftImg8bit']
    data_divion_name = ['train', 'test', 'val']
    output_folder_path_list = []
    # 创建cityscapes组织结构
    print('Clean dataset folder!')
    shutil.rmtree(output_root)
    print('Create new folder:')
    for n in tqdm(cityscapes_folder_list):
        output_folder_path = check_output_path(os.path.join(output_root, n))
        output_folder_path_list.append(output_folder_path)
        for m in tqdm(data_divion_name):
            dataset_division_folder_path = os.path.join(output_folder_path, m)
            check_output_path(dataset_division_folder_path)
            check_output_path(os.path.join(
                dataset_division_folder_path, dataset['dataset_prefix']))

    print('Create annotation file to output folder：')
    for n in tqdm(dataset['temp_divide_file_list'][1:4]):
        dataset_name = os.path.splitext(n.split(os.sep)[-1])[0]
        print('Create annotation file to {} folder:'.format(dataset_name))
        with open(n, 'r') as f:
            pool = multiprocessing.Pool(dataset['workers'])
            for x in tqdm(f.read().splitlines()):
                pool.apply_async(func=F.__dict__[dataset['target_dataset_style']].create_annotation_file,
                                 args=(dataset, file_name_dict, output_folder_path_list,
                                       dataset_name, class_names_dict, x),
                                 error_callback=err_call_back)
            pool.close()
            pool.join()

    return


def coco2017(dataset: dict) -> None:
    """[生成COCO 2017组织格式的数据集]

    Args:
        dataset (dict): [数据集信息字典]
    """

    # 调整image
    print('Clean dataset folder!')
    output_root = check_output_path(
        os.path.join(dataset['target_path'], 'coco2017'))
    shutil.rmtree(output_root)
    output_root = check_output_path(
        os.path.join(dataset['target_path'], 'coco2017'))
    image_output_folder = check_output_path(
        os.path.join(output_root, 'images'))
    annotations_output_folder = check_output_path(
        os.path.join(output_root, 'annotations'))
    # 调整ImageSets
    print('Start copy images:')
    image_list = []
    with open(dataset['temp_divide_file_list'][0], 'r') as f:
        for n in f.readlines():
            image_list.append(n.replace('\n', ''))
    pool = multiprocessing.Pool(dataset['workers'])
    for image_input_path in tqdm(image_list):
        image_output_path = image_input_path.replace(
            dataset['temp_images_folder'], image_output_folder)
        pool.apply_async(func=F.__dict__[dataset['target_dataset_style']].copy_image,
                         args=(image_input_path, image_output_path,), error_callback=err_call_back)
    pool.close()
    pool.join()

    print('Start copy annotations:')
    for root, dirs, files in os.walk(dataset['target_annotations_folder']):
        for n in tqdm(files):
            annotations_input_path = os.path.join(root, n)
            annotations_output_path = annotations_input_path.replace(
            dataset['target_annotations_folder'], annotations_output_folder)
            shutil.copy(annotations_input_path, annotations_output_path)
    return


def cityscapes_val(dataset: dict) -> None:
    """[生成仅包含val集的cityscapes组织结构数据集]

    Args:
        dataset (dict): [数据集信息字典]
    """

    # 获取全量数据编号字典
    file_name_dict = {}
    print('Collect file name dict.')
    for x, n in enumerate(sorted(os.listdir(dataset['temp_images_folder']))):
        file_name = os.path.splitext(n.split(os.sep)[-1])[0]
        file_name_dict[file_name] = x

    output_root = check_output_path(os.path.join(
        dataset['target_path'], 'cityscapes', 'data'))   # 输出数据集文件夹
    cityscapes_folder_list = ['gtFine', 'leftImg8bit']
    data_divion_name = ['train', 'test', 'val']
    output_folder_path_list = []
    # 创建cityscapes组织结构
    print('Clean dataset folder!')
    shutil.rmtree(output_root)
    print('Create new folder:')
    for n in tqdm(cityscapes_folder_list):
        output_folder_path = check_output_path(os.path.join(output_root, n))
        output_folder_path_list.append(output_folder_path)
        for m in tqdm(data_divion_name):
            dataset_division_folder_path = os.path.join(output_folder_path, m)
            check_output_path(dataset_division_folder_path)
            check_output_path(os.path.join(
                dataset_division_folder_path, dataset['dataset_prefix']))

    print('Create annotation file to output folder：')
    for x in tqdm(os.listdir(dataset['temp_images_folder'])):
        file = os.path.splitext(x.split(os.sep)[-1])[0]
        file_out = dataset['dataset_prefix'] + '_000000_' + \
            str(format(file_name_dict[file], '06d'))
        # 调整image
        image_out = file_out + '_leftImg8bit' + \
            '.' + dataset['target_image_form']
        image_path = os.path.join(
            dataset['temp_images_folder'], file + '.' + dataset['target_image_form'])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue

        image_output_path = os.path.join(
            output_folder_path_list[1], 'val', dataset['dataset_prefix'], image_out)
        # 调整annotation
        annotation_out = file_out + '_gtFine_polygons' + \
            '.' + dataset['target_annotation_form']
        annotation_path = os.path.join(
            dataset['target_annotations_folder'], file + '.' + dataset['target_annotation_form'])
        annotation_output_path = os.path.join(
            output_folder_path_list[0], 'val', dataset['dataset_prefix'], annotation_out)
        # 调整annotation为_gtFine_labelIds.png
        labelIds_out = file_out + '_gtFine_labelIds.png'
        labelIds_output_path = os.path.join(
            output_folder_path_list[0], 'val', dataset['dataset_prefix'], labelIds_out)
        # 输出
        shutil.copy(image_path, image_output_path)
        shutil.copy(annotation_path, annotation_output_path)

        img = image.shape
        zeros = np.zeros((img[0], img[1]), dtype=np.uint8)
        cv2.imwrite(labelIds_output_path, zeros)

    return


def cvat_image_1_1(dataset: dict) -> None:

    pass
