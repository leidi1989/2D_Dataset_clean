'''
Description:
Version:
Author: Leidi
Date: 2021-08-11 03:28:09
LastEditors: Leidi
LastEditTime: 2021-11-15 15:20:10
'''
from base.image_base import *
from utils.utils import check_output_path
from annotation.annotation_temp import TEMP_LOAD
from utils.plot import plot_segment_annotation, plot_pick_class_segment_annotation

import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm


def bdd100k(dataset: dict) -> None:
    """[将数据集转换为BDD100K组织格式]

    Args:
        dataset (dict): [数据集信息字典]
    """

    output_root = check_output_path(os.path.join(
        dataset['target_path'], 'target_output'))   # 输出数据集文件夹
    task_list = ['images', 'detect_annotation', 'segment_annotation']
    data_divion_name = ['train', 'val', 'test']
    output_folder_path_list = []
    # 创建cityscapes组织结构
    print('Clean dataset folder!')
    shutil.rmtree(output_root)
    print('Create new folder:')
    for n in tqdm(task_list):
        output_folder_path = check_output_path(os.path.join(output_root, n))
        output_folder_path_list.append(output_folder_path)
        for m in tqdm(data_divion_name):
            dataset_division_folder_path = os.path.join(output_folder_path, m)
            check_output_path(dataset_division_folder_path)

    print('Create annotation file to output folder：')
    for n in tqdm(dataset['temp_divide_file_list'][1:4]):
        dataset_name = os.path.splitext(n.split(os.sep)[-1])[0]
        print('Create annotation file to {} folder:'.format(dataset_name))
        with open(n, 'r') as f:
            for x in tqdm(f.read().splitlines()):
                file = x.split(os.sep)[-1].split('.')[0]
                annotation_load_path = os.path.join(
                    dataset['target_annotations_folder'], file + '.' + dataset['target_annotation_form'])
                # 调整image
                image_out_name = file + '.' + dataset['target_image_form']
                image_load_path = os.path.join(
                    dataset['temp_images_folder'], file + '.' + dataset['target_image_form'])
                image_output_path = os.path.join(
                    output_folder_path_list[0], dataset_name, image_out_name)
                # 调整annotation为detect的annotation路径
                detect_annotation_out_name = file + '.' + \
                    dataset['target_annotation_form']
                detect_annotation_output_path = os.path.join(
                    output_folder_path_list[1], dataset_name, detect_annotation_out_name)
                # 调整annotation为segment的annotation路径
                segment_annotation_out_name = file + '.png'
                segment_annotation_output_path = os.path.join(
                    output_folder_path_list[2], dataset_name, segment_annotation_out_name)
                # 读取temp annotation
                image = TEMP_LOAD(dataset, annotation_load_path)
                # 输出
                shutil.copy(image_load_path, image_output_path)
                shutil.copy(annotation_load_path,
                            detect_annotation_output_path)
                # 调整annotation为语义分割标签，*.png
                plot_segment_annotation(
                    dataset, image, segment_annotation_output_path)

    return


def yolop(dataset: dict) -> None:
    """[将数据集转换为YOLOP组织格式]

    Args:
        dataset (dict): [数据集信息字典]
    """

    output_root = check_output_path(os.path.join(
        dataset['target_path'], 'target_output'))   # 输出数据集文件夹
    task_list = ['images', 'det_annotations',
                 'da_seg_annotations', 'll_seg_annotations']
    data_divion_name = ['train', 'val', 'test']
    output_folder_path_list = []
    # 创建cityscapes组织结构
    print('Clean dataset folder!')
    shutil.rmtree(output_root)
    print('Create new folder:')
    for n in tqdm(task_list):
        output_folder_path = check_output_path(os.path.join(output_root, n))
        output_folder_path_list.append(output_folder_path)
        for m in tqdm(data_divion_name):
            dataset_division_folder_path = os.path.join(output_folder_path, m)
            check_output_path(dataset_division_folder_path)

    print('Create annotation file to output folder：')
    for n in tqdm(dataset['temp_divide_file_list'][1:4]):
        dataset_name = os.path.splitext(n.split(os.sep)[-1])[0]
        print('Create annotation file to {} folder:'.format(dataset_name))
        with open(n, 'r') as f:
            for x in tqdm(f.read().splitlines()):
                file = x.split(os.sep)[-1].split('.')[0]
                annotation_load_path = os.path.join(
                    dataset['target_annotations_folder'], file + '.' + dataset['target_annotation_form'])
                # 调整image
                image_out_name = file + '.' + dataset['target_image_form']
                image_load_path = os.path.join(
                    dataset['temp_images_folder'], file + '.' + dataset['target_image_form'])
                image_output_path = os.path.join(
                    output_folder_path_list[0], dataset_name, image_out_name)
                # 调整annotation为detect的annotation路径
                detect_annotation_out_name = file + '.' + \
                    dataset['target_annotation_form']
                detect_annotation_output_path = os.path.join(
                    output_folder_path_list[1], dataset_name, detect_annotation_out_name)
                # 调整annotation为segment的annotation路径
                segment_annotation_out_name = file + '.png'
                segment_annotation_output_path = os.path.join(
                    output_folder_path_list[2], dataset_name, segment_annotation_out_name)
                # 调整annotation为lane segment的annotation路径
                lane_segment_annotation_out_name = file + '.png'
                lane_segment_annotation_output_path = os.path.join(
                    output_folder_path_list[3], dataset_name, lane_segment_annotation_out_name)
                # 读取temp annotation
                image = TEMP_LOAD(dataset, annotation_load_path)
                # 输出图片
                shutil.copy(image_load_path, image_output_path)
                # 输出目标检测标签
                # if 0 != len(image.true_box_list):
                #     shutil.copy(annotation_load_path,
                #                 detect_annotation_output_path)
                shutil.copy(annotation_load_path,
                                detect_annotation_output_path)
                # 调整annotation为路面语义分割标签，*.png
                road_class_list = ['road']
                road_color = 127
                plot_pick_class_segment_annotation(
                    dataset, image, segment_annotation_output_path, road_class_list, road_color)
                # 调整annotation为车道线语义分割标签，*.png
                lane_class_list = ['lane']
                lane_color = 255
                plot_pick_class_segment_annotation(
                    dataset, image, lane_segment_annotation_output_path, lane_class_list, lane_color)

    return


# def CITYSCAPES_FRAMEWORK(dataset: dict) -> None:
#     """[生成cityscapes组织结构数据集]

#     Args:
#         dataset (dict): [数据集信息字典]
#     """

#     # 官方数值
#     # colors = [
#     #     [128, 64, 128], [244, 35, 232], [70, 70, 70], [
#     #         102, 102, 156], [190, 153, 153],
#     #     [153, 153, 153], [250, 170, 30], [220, 220, 0], [
#     #         107, 142, 35],  [152, 251, 152],
#     #     [0, 130, 180],  [220, 20, 60],  [
#     #         255, 0, 0],  [0, 0, 142],     [0, 0, 70],
#     #     [0, 60, 100],   [0, 80, 100],   [0, 0, 230],  [119, 11, 32], [0, 0, 0]]
#     # label_colours = dict(zip(range(19), colors))
#     # void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
#     # valid_classes = [7, 8, 11, 12, 13, 17, 19, 20,
#     #                  21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
#     # class_map = dict(zip(valid_classes, range(19)))
#     # class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
#     #                'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
#     #                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
#     #                'motorcycle', 'bicycle']
#     # ignore_index = 255
#     # num_classes_ = 19
#     # class_weights_ = np.array([0.05570516, 0.32337477, 0.08998544, 1.03602707, 1.03413147, 1.68195437,
#     #                            5.58540548, 3.56563995, 0.12704978, 1., 0.46783719, 1.34551528,
#     #                            5.29974114, 0.28342531, 0.9396095, 0.81551811, 0.42679146, 3.6399074,
#     #                            2.78376194], dtype=float)
#     # labelIds
#     class_names_dict = {'unlabeled': 0,
#                         'egovehicle': 1,
#                         'rectificationborder': 2,
#                         'outofroi': 3,
#                         'static': 4,
#                         'dynamic': 5,
#                         'ground': 6,
#                         'road': 7,
#                         'sidewalk': 8,
#                         'parking': 9,
#                         'railtrack': 10,
#                         'building': 11,
#                         'wall': 12,
#                         'fence': 13,
#                         'guardrail': 14,
#                         'bridge': 15,
#                         'tunnel': 16,
#                         'pole': 17,
#                         'polegroup': 18,
#                         'trafficlight': 19,
#                         'trafficsign': 20,
#                         'vegetation': 21,
#                         'terrain': 22,
#                         'sky': 23,
#                         'person': 24,
#                         'rider': 25,
#                         'car': 26,
#                         'truck': 27,
#                         'bus': 28,
#                         'caravan': 29,
#                         'trailer': 30,
#                         'train': 31,
#                         'motorcycle': 32,
#                         'bicycle': 33,
#                         'licenseplate': -1,
#                         }
#     # trainid
#     # class_names_dict = {'unlabeled': 0,
#     #                     'egovehicle': 1,
#     #                     'rectificationborder': 2,
#     #                     'outofroi': 3,
#     #                     'static': 4,
#     #                     'dynamic': 5,
#     #                     'ground': 6,
#     #                     'road': 7,
#     #                     'sidewalk': 8,
#     #                     'parking': 9,
#     #                     'railtrack': 10,
#     #                     'building': 11,
#     #                     'wall': 12,
#     #                     'fence': 13,
#     #                     'guardrail': 14,
#     #                     'bridge': 15,
#     #                     'tunnel': 16,
#     #                     'pole': 17,
#     #                     'polegroup': 18,
#     #                     'trafficlight': 19,
#     #                     'trafficsign': 20,
#     #                     'vegetation': 21,
#     #                     'terrain': 22,
#     #                     'sky': 23,
#     #                     'person': 24,
#     #                     'rider': 25,
#     #                     'car': 26,
#     #                     'truck': 27,
#     #                     'bus': 28,
#     #                     'caravan': 29,
#     #                     'trailer': 30,
#     #                     'train': 31,
#     #                     'motorcycle': 32,
#     #                     'bicycle': 33,
#     #                     'licenseplate': -1,
#     #                     }

#     # 获取全量数据编号字典
#     file_name_dict = {}
#     print('Collect file name dict.')
#     with open(dataset['temp_divide_file_list'][0], 'r') as f:
#         for x, n in enumerate(f.read().splitlines()):
#             file_name = n.split(os.sep)[-1].split('.')[0]
#             file_name_dict[file_name] = x
#         f.close()

#     output_root = check_output_path(os.path.join(
#         dataset['target_path'], 'cityscapes', 'data'))   # 输出数据集文件夹
#     cityscapes_folder_list = ['gtFine', 'leftImg8bit']
#     data_divion_name = ['train', 'test', 'val']
#     output_folder_path_list = []
#     # 创建cityscapes组织结构
#     print('Clean dataset folder!')
#     shutil.rmtree(output_root)
#     print('Create new folder:')
#     for n in tqdm(cityscapes_folder_list):
#         output_folder_path = check_output_path(os.path.join(output_root, n))
#         output_folder_path_list.append(output_folder_path)
#         for m in tqdm(data_divion_name):
#             dataset_division_folder_path = os.path.join(output_folder_path, m)
#             check_output_path(dataset_division_folder_path)
#             check_output_path(os.path.join(
#                 dataset_division_folder_path, dataset['dataset_prefix']))

#     print('Create annotation file to output folder：')
#     for n in tqdm(dataset['temp_divide_file_list'][1:4]):
#         dataset_name = n.split(os.sep)[-1].split('.')[0]
#         print('Create annotation file to {} folder:'.format(dataset_name))
#         with open(n, 'r') as f:
#             for x in tqdm(f.read().splitlines()):
#                 file = x.split(os.sep)[-1].split('.')[0]
#                 file_out = dataset['dataset_prefix'] + '_000000_' + \
#                     str(format(file_name_dict[file], '06d'))
#                 # 调整image
#                 image_out = file_out + '_leftImg8bit' + \
#                     '.' + dataset['target_image_form']
#                 image_path = os.path.join(
#                     dataset['temp_images_folder'], file + '.' + dataset['target_image_form'])
#                 image_output_path = os.path.join(
#                     output_folder_path_list[1], dataset_name, dataset['dataset_prefix'], image_out)
#                 # 调整annotation
#                 annotation_out = file_out + '_gtFine_polygons' + \
#                     '.' + dataset['target_annotation_form']
#                 annotation_path = os.path.join(
#                     dataset['target_annotations_folder'], file + '.' + dataset['target_annotation_form'])
#                 annotation_output_path = os.path.join(
#                     output_folder_path_list[0], dataset_name, dataset['dataset_prefix'], annotation_out)
#                 # 调整annotation为_gtFine_labelIds.png
#                 image = TEMP_LOAD(dataset, annotation_path)
#                 labelIds_out = file_out + '_gtFine_labelIds.png'
#                 labelIds_output_path = os.path.join(
#                     output_folder_path_list[0], dataset_name, dataset['dataset_prefix'], labelIds_out)
#                 # 输出
#                 shutil.copy(image_path, image_output_path)
#                 shutil.copy(annotation_path, annotation_output_path)

#                 zeros = np.zeros((image.height, image.width), dtype=np.uint8)
#                 if len(image.true_segmentation_list):
#                     for seg in image.true_segmentation_list:
#                         class_color = class_names_dict[seg.clss]
#                         points = np.array(seg.segmentation)
#                         zeros_mask = cv2.fillPoly(
#                             zeros, pts=[points], color=class_color)
#                         cv2.imwrite(labelIds_output_path, zeros_mask)
#                 else:
#                     cv2.imwrite(labelIds_output_path, zeros)

#     # 调整annotation为_gtFine_labelIds.png
#     print('Create labelIds.png to output folder：')
#     for n in tqdm(dataset['temp_divide_file_list'][1:4]):
#         dataset_name = n.split(os.sep)[-1].split('.')[0]
#         print('Create labelTrainIds.png to {} folder:'.format(dataset_name))
#         with open(n, 'r') as f:
#             for x in tqdm(f.read().splitlines()):
#                 file = x.split(os.sep)[-1].split('.')[0] + \
#                     '.' + dataset['target_annotation_form']
#                 file_name_list = x.split(
#                     os.sep)[-1].split('.')[0].split(dataset['prefix_delimiter'])
#                 file_name = file_name_list[0] + '_999999_9' + file_name_list[1]
#                 file_out = file_name + '_gtFine_labelIds' + \
#                     '.' + dataset['target_image_form']

#     return


# def PASCAL_VOC_FRAMEWORK(dataset: dict) -> None:
#     """[调整数据集组织格式为PASCAL VOC数据集组织格式]

#     Args:
#         dataset (dict): [数据集信息字典]
#     """

#     # 调整image
#     print('Change images folder.')
#     image_output_path = check_output_path(
#         os.path.join(dataset['target_path'], 'JPEGImages'))
#     os.rename(dataset['source_images_folder'], image_output_path)
#     # 调整ImageSets
#     print('Change ImageSets folder:')
#     imagesets_path = check_output_path(
#         os.path.join(dataset['target_path'], 'Dataset_infomations'))
#     for n in tqdm(dataset['temp_divide_file_list']):
#         print('Update {}'.format(n.split(os.sep)[-1]))
#         output_list = []
#         with open(n, 'r') as f:
#             annotation_path_list = f.read().splitlines()
#             for m in annotation_path_list:
#                 output_list.append(m.replace('source_images', 'JPEGImages'))
#             f.close()
#         with open(os.path.join(imagesets_path, n.split(os.sep)[-1]), 'w') as f:
#             for m in output_list:
#                 f.write('%s\n' % m)
#             f.close()
#         os.remove(n)
#     # 调整文件夹
#     print('Update folder.')
#     shutil.move(os.path.join(dataset['temp_informations_folder'],
#                              'Main'), os.path.join(imagesets_path, 'Main'))
#     os.rename(dataset['temp_information_folder'], os.path.join(
#         dataset['output_path'], 'Dataset_information'))

#     return


# def COCO_2017_FRAMEWORK(dataset: dict):
#     """[生成COCO 2017组织格式的数据集]

#     Args:
#         dataset (dict): [数据集信息字典]
#     """

#     # 调整image
#     print('Change images folder.')
#     image_output_path = check_output_path(
#         os.path.join(dataset['output_path'], 'images'))
#     os.rename(dataset['temp_images_folder'], image_output_path)

#     # 调整ImageSets
#     print('Change ImageSets folder:')
#     imagesets_path = check_output_path(
#         os.path.join(dataset['output_path'], 'ImageSets'))
#     for n in dataset['temp_divide_file_list']:
#         print('Update {} '.format(n.split(os.sep)[-1]))
#         output_list = []
#         with open(n, 'r') as f:
#             annotation_path_list = f.read().splitlines()
#             for m in tqdm(annotation_path_list):
#                 output_list.append(m.replace('temp_images', 'images'))
#             f.close()
#         with open(os.path.join(imagesets_path, n.split(os.sep)[-1]), 'w') as f:
#             for m in output_list:
#                 f.write('%s\n' % m)
#             f.close()
#         os.remove(n)
#     # 调整文件夹
#     print('Update folder.')
#     os.rename(dataset['temp_information_folder'], os.path.join(
#         dataset['output_path'], 'Dataset_information'))

#     return


# def CITYSCAPESVAL_FRAMEWORK(dataset: dict) -> None:
#     """[生成仅包含val集的cityscapes组织结构数据集]

#     Args:
#         dataset (dict): [数据集信息字典]
#     """

#     # 获取全量数据编号字典
#     file_name_dict = {}
#     print('Collect file name dict.')
#     for x, n in enumerate(sorted(os.listdir(dataset['temp_images_folder']))):
#         file_name = n.split(os.sep)[-1].split('.')[0]
#         file_name_dict[file_name] = x

#     output_root = check_output_path(os.path.join(
#         dataset['target_path'], 'cityscapes', 'data'))   # 输出数据集文件夹
#     cityscapes_folder_list = ['gtFine', 'leftImg8bit']
#     data_divion_name = ['train', 'test', 'val']
#     output_folder_path_list = []
#     # 创建cityscapes组织结构
#     print('Clean dataset folder!')
#     shutil.rmtree(output_root)
#     print('Create new folder:')
#     for n in tqdm(cityscapes_folder_list):
#         output_folder_path = check_output_path(os.path.join(output_root, n))
#         output_folder_path_list.append(output_folder_path)
#         for m in tqdm(data_divion_name):
#             dataset_division_folder_path = os.path.join(output_folder_path, m)
#             check_output_path(dataset_division_folder_path)
#             check_output_path(os.path.join(
#                 dataset_division_folder_path, dataset['dataset_prefix']))

#     print('Create annotation file to output folder：')
#     for x in tqdm(os.listdir(dataset['temp_images_folder'])):
#         file = x.split(os.sep)[-1].split('.')[0]
#         file_out = dataset['dataset_prefix'] + '_000000_' + \
#             str(format(file_name_dict[file], '06d'))
#         # 调整image
#         image_out = file_out + '_leftImg8bit' + \
#             '.' + dataset['target_image_form']
#         image_path = os.path.join(
#             dataset['temp_images_folder'], file + '.' + dataset['target_image_form'])
#         image_output_path = os.path.join(
#             output_folder_path_list[1], 'val', dataset['dataset_prefix'], image_out)
#         # 调整annotation
#         annotation_out = file_out + '_gtFine_polygons' + \
#             '.' + dataset['target_annotation_form']
#         annotation_path = os.path.join(
#             dataset['target_annotations_folder'], file + '.' + dataset['target_annotation_form'])
#         annotation_output_path = os.path.join(
#             output_folder_path_list[0], 'val', dataset['dataset_prefix'], annotation_out)
#         # 调整annotation为_gtFine_labelIds.png
#         labelIds_out = file_out + '_gtFine_labelIds.png'
#         labelIds_output_path = os.path.join(
#             output_folder_path_list[0], 'val', dataset['dataset_prefix'], labelIds_out)
#         # 输出
#         shutil.copy(image_path, image_output_path)
#         shutil.copy(annotation_path, annotation_output_path)

#         image = cv2.imread(image_path)
#         img = image.shape
#         zeros = np.zeros((img[0], img[1]), dtype=np.uint8)
#         cv2.imwrite(labelIds_output_path, zeros)

#     return
