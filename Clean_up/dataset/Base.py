'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 11:00:30
LastEditors: Leidi
LastEditTime: 2022-01-07 11:46:41
'''
import os
import time
import yaml
import argparse
import multiprocessing

from utils.utils import *
from input import source_dataset
from out import framework_update
from base.check_base import check
from annotation import annotation_load
from annotation import annotation_output
from base.dataset_characteristic import *
from base.information_base import information


class Dataset_Base:
    """[数据集基础类]
    """
    def __init__(self, opt) -> None:
        # load dataset config file
        dataset_config = yaml.load(
            open(opt.config, 'r', encoding="utf-8"), Loader=yaml.FullLoader)
        
        # source dataset
        self.source_path = check_input_path(dataset_config['root'])
        self.source_dataset_style = dataset_config['srstyle']
        self.source_image_form = dataset_file_form[dataset_config['srstyle']]['image']
        self.source_images_folder = check_output_path(
            os.path.join(dataset_config['target'], 'source_images'))
        self.source_annotation_form = dataset_file_form[dataset_config['srstyle']]['annotation']
        self.source_annotations_folder = check_output_path(
            os.path.join(dataset_config['target'], 'source_annotations'))
        self.source_class_list = get_class_list(dataset_config['classes'])
        
        # prefix
        self.prefix_delimiter = dataset_config['delimiter']
        self.file_prefix = check_prefix(dataset_config['prefix'], dataset_config['delimiter'])
        
        # object classes
        self.modify_class_dict = get_modify_class_dict(dataset_config['modifyclassfile'])
        self.class_list_new = get_new_class_names_list(get_class_list(dataset_config['classes']), get_modify_class_dict(dataset_config['modifyclassfile']))
        self.class_pixel_distance_dict = get_class_pixel_limit(dataset_config['classpixeldistance'])
        
        # temp dataset
        self.temp_image_form = dataset_file_form[dataset_config['tarstyle']]['image']
        self.temp_annotation_form = temp_form['annotation']
        self.temp_images_folder = check_output_path(os.path.join(dataset_config['target'], 'source_images'))
        self.temp_annotations_folder = check_output_path(os.path.join(dataset_config['target'], temp_arch['annotation']))
        self.temp_informations_folder = check_output_path(os.path.join(dataset_config['target'], 'temp_infomations'))
        self.temp_divide_file_list = [
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
        self.temp_file_name_list=temp_file_name(self.temp_annotations_folder)
        self.temp_annotation_path_list=temp_annotation_path_list(self.temp_annotations_folder)
        
        # target dataset
        self.target_path = check_output_path(dataset_config['target'])
        self.target_dataset_style = dataset_config['tarstyle']
        self.target_image_form = dataset_file_form[dataset_config['tarstyle']]['image']
        self.target_annotation_form = dataset_file_form[dataset_config['tarstyle']]['annotation']
        self.target_annotations_folder = check_output_path(
            os.path.join(dataset_config['target'], 'target_annotations'))
        
        # temp dataset information
        self.total_file_name_path = total_file(self.temp_informations_folder)
        self.proportion = tuple(float(x)
                                for x in (dataset_config['proportion'].split(',')))
        self.image_resolution_median = None
        self.anchor_box_cluster = dataset_config['anchorboxcluster']
        
        # target check
        self.target_annotation_check_count = dataset_config['check']
        self.target_annotation_check_mask = dataset_config['mask']
        self.check_annotation_output_folder = check_output_path(os.path.join(
            self.temp_informations_folder, 'check_annotation'))
        self.check_file_name_list = None
        self.check_images_list = None
        
        # others
        self.workers = opt.workers
        self.debug = dataset_config['debug']
        