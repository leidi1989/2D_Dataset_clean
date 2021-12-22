'''
Description:
Version:
Author: Leidi
Date: 2021-08-11 03:28:09
LastEditors: Leidi
LastEditTime: 2021-12-22 15:57:28
'''
import os
import shutil
from tqdm import tqdm
import multiprocessing

from base.image_base import *
import out.framwork_update_function as F
from annotation.annotation_temp import TEMP_LOAD
from utils.utils import check_output_path, err_call_back
from utils.plot import plot_segment_annotation, plot_pick_class_segment_annotation


def bdd100k(dataset: dict) -> None:
    """[将数据集转换为BDD100K组织格式]

    Args:
        dataset (dict): [数据集信息字典]
    """

    output_root = check_output_path(os.path.join(
        dataset['target_path'], 'bdd100k'))   # 输出数据集文件夹
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

    print('Create annotation file to output folder:')
    for n in tqdm(dataset['temp_divide_file_list'][1:4]):
        dataset_name = os.path.splitext(n.split(os.sep)[-1])[0]
        print('Create annotation file to {} folder:'.format(dataset_name))
        with open(n, 'r') as f:
            for x in tqdm(f.read().splitlines()):
                file = os.path.splitext(x.split(os.sep)[-1])[0]
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
        dataset['target_path'], 'yolop'))   # 输出数据集文件夹
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

    print('Create annotation file to output folder:')
    for n in tqdm(dataset['temp_divide_file_list'][1:4]):
        dataset_name = os.path.splitext(n.split(os.sep)[-1])[0]
        print('Create annotation file to {} folder:'.format(dataset_name))
        with open(n, 'r') as f:
            for x in tqdm(f.read().splitlines()):
                file = os.path.splitext(x.split(os.sep)[-1])[0]
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
