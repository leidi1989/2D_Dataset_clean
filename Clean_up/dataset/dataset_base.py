'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 11:00:30
LastEditors: Leidi
LastEditTime: 2022-01-10 11:39:04
'''
import os
import yaml

from utils.utils import *
from .dataset_characteristic import *
from .information_base import information


class Dataset_Base:
    """[数据集基础类]
    """

    def __init__(self, dataset_config) -> None:

        # Dataset_style
        self.dataset_style = dataset_config['Dataset_style']

        # Source_dataset
        self.dataset_input_folder = check_input_path(
            dataset_config['Dataset_input_folder'])
        self.source_dataset_style = dataset_config['Source_dataset_style']
        self.source_dataset_image_form = DATASET_FILE_FORM[
            dataset_config['Source_dataset_style']]['image']
        self.source_dataset_images_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'], 'source_dataset_images'))
        self.source_dataset_annotation_form = DATASET_FILE_FORM[
            dataset_config['Source_dataset_style']]['annotation']
        self.source_dataset_annotations_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'], 'source_dataset_annotations'))
        self.task_list = list()
        self.source_dataset_class_list = list()
        self.modify_class_dict_list = list()

        # File_prefix
        self.file_prefix_delimiter = dataset_config['File_prefix_delimiter']
        self.file_prefix = check_prefix(
            dataset_config['File_prefix'], dataset_config['File_prefix_delimiter'])

        # temp dataset
        self.temp_image_form = DATASET_FILE_FORM[dataset_config['Target_dataset_style']]['image']
        self.temp_annotation_form = TEMP_FORM['annotation']
        self.temp_images_folder = check_output_path(os.path.join(
            dataset_config['Dataset_output_folder'], 'source_dataset_images'))
        self.temp_annotations_folder = check_output_path(os.path.join(
            dataset_config['Dataset_output_folder'], TEMP_ARCH['annotation']))
        self.temp_informations_folder = check_output_path(os.path.join(
            dataset_config['Dataset_output_folder'], 'temp_infomations'))
        self.temp_divide_file_list = [
            os.path.join(
                os.path.join(dataset_config['Dataset_output_folder'], 'temp_infomations'), 'total.txt'),
            os.path.join(
                os.path.join(dataset_config['Dataset_output_folder'], 'temp_infomations'), 'train.txt'),
            os.path.join(
                os.path.join(dataset_config['Dataset_output_folder'], 'temp_infomations'), 'test.txt'),
            os.path.join(
                os.path.join(dataset_config['Dataset_output_folder'], 'temp_infomations'), 'val.txt'),
            os.path.join(
                os.path.join(dataset_config['Dataset_output_folder'], 'temp_infomations'), 'redund.txt')
        ]
        self.temp_annotation_name_list = get_temp_annotations_name_list(
            self.temp_annotations_folder)
        self.temp_annotations_path_list = temp_annotations_path_list(
            self.temp_annotations_folder)

        # target dataset
        self.dataset_output_folder = check_output_path(
            dataset_config['Dataset_output_folder'])
        self.target_dataset_style = dataset_config['Target_dataset_style']
        self.target_dataset_image_form = DATASET_FILE_FORM[
            dataset_config['Target_dataset_style']]['image']
        self.target_dataset_annotation_form = DATASET_FILE_FORM[
            dataset_config['Target_dataset_style']]['annotation']
        self.target_dataset_annotations_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'], 'target_dataset_annotations'))
        self.target_dataset_class_list = list()

        # temp dataset information
        self.total_file_name_path = total_file(
            self.temp_informations_folder)
        self.target_dataset_divide_proportion = tuple(float(x)
                                                      for x in (dataset_config['Target_dataset_divide_proportion'].split(',')))

        # target check
        self.target_dataset_annotations_check_count = dataset_config[
            'Target_dataset_check_annotations_count']
        self.target_dataset_annotation_check_output_folder = check_output_path(os.path.join(
            self.temp_informations_folder, 'check_annotation'))
        self.target_dataset_check_file_name_list = None
        self.target_dataset_check_images_list = None
        self.target_dataset_annotation_check_mask = dataset_config[
            'Target_dataset_check_annotations_output_as_mask']

        # others
        self.workers = dataset_config['workers']
        self.debug = dataset_config['debug']

        # task and class

        for task, \
            source_dataset_class, \
            modify_class_file in zip([x for x in dataset_config['Task_and_class_config']
                                      ['Task'].values()],
                                     [y for y in dataset_config['Task_and_class_config']
                                         ['Source_dataset_class_file_path'].values()],
                                     [z for z in dataset_config['Task_and_class_config']
                                      ['Modify_class_file_path'].values()]):
            source_dataset_class = get_class_list(source_dataset_class)
            modify_class_dict = get_modify_class_dict(modify_class_file)
            target_dataset_class = get_new_class_names_list(
                source_dataset_class, modify_class_dict)
            self.task_list.append(task)
            self.source_dataset_class_list.append(source_dataset_class)
            self.modify_class_dict_list.append(modify_class_dict)
            self.target_dataset_class_list.append(target_dataset_class)

    def source_dataset_copy_image_and_annotation(self):
        # print('\nStart source dataset copy image and annotation:')
        raise NotImplementedError("ERROR: func not implemented!")

    def source_dataset_copy_image(self):
        # print('\nStart source dataset copy image and annotation:')
        raise NotImplementedError("ERROR: func not implemented!")

    def source_dataset_copy_annotation(self):
        # print('\nStart source dataset copy image and annotation:')
        raise NotImplementedError("ERROR: func not implemented!")

    def transform_to_temp_dataset(self):
        # print('\nStart transform to temp dataset:')
        raise NotImplementedError("ERROR: func not implemented!")

    def get_dataset_information(self):
        # print('\nStart get temp dataset information:')
        information(self)

    def transform_to_target_dataset():
        # print('\nStart transform to target dataset:')
        raise NotImplementedError("ERROR: func not implemented!")

    def build_target_dataset_folder():
        # print('\nStart build target dataset folder:')
        raise NotImplementedError("ERROR: func not implemented!")
