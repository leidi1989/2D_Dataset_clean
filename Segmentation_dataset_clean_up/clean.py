'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:45:50
LastEditors: Leidi
LastEditTime: 2021-11-24 15:11:40
'''
from utils.utils import *
from input import source_dataset
from out import framework_update
from base.check_base import check
from annotation import annotation_load
from annotation import annotation_output
from base.dataset_characteristic import *
from base.information_base import information

import os
import time
import yaml
import argparse
import multiprocessing


def main(dataset_info: dict) -> None:
    """[数据集清理]

    Args:
        dataset_info (dict): [数据集信息字典]
    """

    print('\nStart copy images and annotations:')
    source_dataset.__dict__[dataset_info['source_dataset_stype']](dataset_info)

    print('\nStart to transform source annotation to temp annotation:')
    annotation_load.__dict__[
        dataset_info['source_dataset_stype']](dataset_info)

    print('\nStart to analyze dataset:')
    information(dataset_info)

    print('\nStart output temp dataset annotations to target annotations:')
    dataset_info['temp_annotation_path_list'] = temp_annotation_path_list(
        dataset_info['temp_annotations_folder'])
    annotation_output.__dict__[
        dataset_info['target_dataset_style']](dataset_info)

    print('\nStart check target annotations:')
    check(dataset_info)

    print('\nStart update framework:')
    framework_update.__dict__[
        dataset_info['target_dataset_style']](dataset_info)

    return


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='clean.py')
    parser.add_argument('--config', '--c', dest='config', default=r'/home/leidi/hy_program/2D_Dataset_clean/Segmentation_dataset_clean_up/config/default.yaml',
                        type=str, help='dataset config file path')
    parser.add_argument('--workers', '--w', dest='workers', default=multiprocessing.cpu_count(),
                        type=int, help='maximum number of dataloader workers(multiprocessing.cpu_count())')

    opt = parser.parse_args()

    dataset_config = yaml.load(
        open(opt.config, 'r', encoding="utf-8"), Loader=yaml.FullLoader)

    source_path = check_input_path(dataset_config['root'])
    source_dataset_style = dataset_config['srstyle']
    source_image_form = dataset_file_form[dataset_config['srstyle']]['image']
    source_images_folder = check_output_path(
        os.path.join(dataset_config['target'], 'source_images'))
    source_annotation_form = dataset_file_form[dataset_config['srstyle']]['annotation']
    source_annotations_folder = check_output_path(
        os.path.join(dataset_config['target'], 'source_annotations'))
    source_class_list = get_class_list(dataset_config['classes'])
    prefix_delimiter = dataset_config['delimiter']
    file_prefix = check_prefix(
        dataset_config['prefix'], dataset_config['delimiter'])
    modify_class_dict = get_modify_class_dict(
        dataset_config['modifyclassfile'])
    class_list_new = get_new_class_names_list(get_class_list(
        dataset_config['classes']), get_modify_class_dict(dataset_config['modifyclassfile']))
    temp_image_form = dataset_file_form[dataset_config['tarstyle']]['image']
    temp_annotation_form = temp_form['annotation']
    temp_images_folder = check_output_path(
        os.path.join(dataset_config['target'], 'source_images'))
    temp_annotations_folder = check_output_path(
        os.path.join(dataset_config['target'], temp_arch['annotation']))
    temp_informations_folder = check_output_path(
        os.path.join(dataset_config['target'], 'temp_infomations'))
    temp_divide_file_list = [
        os.path.join(
            os.path.join(dataset_config['target'], 'temp_infomations'), 'total.txt'),
        os.path.join(
            os.path.join(dataset_config['target'], 'temp_infomations'), 'train.txt'),
        os.path.join(
            os.path.join(dataset_config['target'], 'temp_infomations'), 'test.txt'),
        os.path.join(
            os.path.join(dataset_config['target'], 'temp_infomations'), 'val.txt'),
        os.path.join(
            os.path.join(dataset_config['target'], 'temp_infomations'), 'redund.txt')
    ]
    class_pixel_distance_dict = get_class_pixel_limit(
        dataset_config['classpixeldistance'])
    target_path = check_output_path(dataset_config['target'])
    target_dataset_style = dataset_config['tarstyle']
    target_image_form = dataset_file_form[dataset_config['tarstyle']]['image']
    target_annotation_form = dataset_file_form[dataset_config['tarstyle']]['annotation']
    target_annotations_folder = check_output_path(
        os.path.join(dataset_config['target'], 'target_annotations'))
    proportion = tuple(float(x)
                       for x in (dataset_config['proportion'].split(',')))
    target_annotation_check_count = dataset_config['check']
    target_annotation_check_mask = dataset_config['mask']
    check_annotation_output_folder = check_output_path(os.path.join(
        temp_informations_folder, 'check_annotation'))
    workers = opt.workers
    debug = dataset_config['debug']

    dataset_info = {
        # 源数据集路径、类别，其图片、annotation格式，类别列表文件
        'source_path': source_path,
        'source_dataset_stype': source_dataset_style,
        'source_image_form': source_image_form,
        'source_images_folder': source_images_folder,
        'source_annotation_form': source_annotation_form,
        'source_annotations_folder': source_annotations_folder,
        'source_class_list': source_class_list,
        # 文件前缀分隔符、文件前缀
        'dataset_prefix': dataset_config['prefix'],
        'prefix_delimiter': prefix_delimiter,
        'file_prefix': file_prefix,
        # 修改类别字典、新类别列表
        'modify_class_dict': modify_class_dict,
        'class_list_new': class_list_new,
        # 暂存数据集路径、类别，其图片、annotation格式
        'temp_image_form': temp_image_form,
        'temp_annotation_form': temp_annotation_form,
        'temp_images_folder': temp_images_folder,
        'temp_annotations_folder': temp_annotations_folder,
        'temp_informations_folder': temp_informations_folder,
        'temp_divide_file_list': temp_divide_file_list,
        'temp_file_name_list': temp_file_name(temp_annotations_folder),
        'temp_annotation_path_list': temp_annotation_path_list(temp_annotations_folder),
        'proportion': proportion,
        # 不同类别像素大小真实框选择
        'class_pixel_distance_dict': class_pixel_distance_dict,
        # 目标数据集路径类别、图片、annotation格式
        'target_path': target_path,
        'target_dataset_style': target_dataset_style,
        'target_image_form': target_image_form,
        'target_annotation_form': target_annotation_form,
        'target_annotations_folder': target_annotations_folder,
        'total_file_name_path': total_file(temp_informations_folder),
        'target_annotation_check_count': target_annotation_check_count,
        'check_file_name_list': None,
        'check_images_list': None,
        'target_annotation_check_mask': target_annotation_check_mask,
        'check_annotation_output_folder': check_annotation_output_folder,
        'image_resolution_median': None,
        'anchor_box_cluster': dataset_config['anchorboxcluster'],
        'workers': opt.workers,
        'debug': debug
    }

    main(dataset_info)
