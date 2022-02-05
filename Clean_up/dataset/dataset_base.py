'''
Description:
Version:
Author: Leidi
Date: 2022-01-07 11:00:30
LastEditors: Leidi
LastEditTime: 2022-02-06 00:33:22
'''
import dataset
from utils.utils import *
from .dataset_characteristic import *
from .image_base import IMAGE, OBJECT

import os
import cv2
import math
import shutil
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='AR PL UMing CN')
plt.switch_backend('agg')


class Dataset_Base:
    """[数据集基础类]
    """

    def __init__(self, dataset_config: dict) -> None:
        """[数据集基础类]

        Args:
            dataset_config (dict): [数据集配置信息字典]
        """

        print('Start dataset instance initialize:')
        # Source_dataset
        self.dataset_input_folder = check_input_path(
            dataset_config['Dataset_input_folder'])
        self.source_dataset_style = dataset_config['Source_dataset_style']
        self.source_dataset_image_form_list = None
        self.source_dataset_images_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'], 'source_dataset_images'))
        self.source_dataset_annotation_form = None
        self.source_dataset_annotations_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'], 'source_dataset_annotations'))
        self.source_dataset_image_count = None
        self.source_dataset_annotation_count = None
        self.task_dict = dict()

        # File_prefix
        self.file_prefix_delimiter = dataset_config['File_prefix_delimiter']
        self.file_prefix = check_prefix(
            dataset_config['File_prefix'], dataset_config['File_prefix_delimiter'])

        # temp dataset
        self.temp_image_form = TARGET_DATASET_FILE_FORM[dataset_config['Target_dataset_style']]['image']
        self.temp_annotation_form = 'json'
        self.temp_images_folder = check_output_path(os.path.join(
            dataset_config['Dataset_output_folder'], 'source_dataset_images'))
        self.temp_annotations_folder = check_output_path(os.path.join(
            dataset_config['Dataset_output_folder'], 'temp_annotations'))
        self.temp_informations_folder = check_output_path(os.path.join(
            dataset_config['Dataset_output_folder'], 'temp_infomations'))
        self.temp_sample_statistics_folder = check_output_path(
            os.path.join(self.temp_informations_folder, 'sample_statistics'))
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
        self.temp_set_name_list = ['total_distibution.txt', 'train_distibution.txt',
                                   'val_distibution.txt', 'test_distibution.txt',
                                   'redund_distibution.txt']
        self.temp_annotation_name_list = get_temp_annotations_name_list(
            self.temp_annotations_folder)
        self.temp_annotations_path_list = temp_annotations_path_list(
            self.temp_annotations_folder)

        # target dataset
        self.dataset_output_folder = check_output_path(
            dataset_config['Dataset_output_folder'])
        self.target_dataset_style = dataset_config['Target_dataset_style']
        self.target_dataset_image_form = TARGET_DATASET_FILE_FORM[
            dataset_config['Target_dataset_style']]['image']
        self.target_dataset_annotation_form = TARGET_DATASET_FILE_FORM[
            dataset_config['Target_dataset_style']]['annotation']
        self.target_dataset_annotations_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'], 'target_dataset_annotations'))

        # temp dataset information
        self.total_file_name_path = total_file(
            self.temp_informations_folder)
        self.target_dataset_divide_proportion = tuple(float(x)
                                                      for x in (dataset_config['Target_dataset_divide_proportion'].split(',')))
        self.temp_divide_file_annotation_path = []
        # 声明set类别计数字典列表顺序为ttvt
        self.temp_divide_count_dict_list_dict = {}
        # 声明set类别计数字典列表顺序为ttvt
        self.temp_divide_proportion_dict_list_dict = {}
        self.temp_merge_class_list = {'Merge_source_dataset_class_list': [],
                                      'Merge_target_dataset_class_list': []}

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
        self.task_convert = {}
        for task, task_info in dataset_config['Task_and_class_config'].items():
            source_dataset_class = get_class_list(
                task_info['Source_dataset_class_file_path'])
            modify_class_dict = get_modify_class_dict(
                task_info['Modify_class_file_path'])
            target_dataset_class = get_new_class_names_list(
                source_dataset_class, modify_class_dict)
            object_pixel_limit_dict = get_class_pixel_limit(
                task_info['Target_each_class_object_pixel_limit_file_path'])
            need_conver = task_info['Need_convert']
            self.task_dict[task] = {'Source_dataset_class': source_dataset_class,
                                    'Modify_class_dict': modify_class_dict,
                                    'Target_dataset_class': target_dataset_class,
                                    'Target_object_pixel_limit_dict': object_pixel_limit_dict,
                                    }
            self.task_convert.update({task: need_conver})
            self.temp_merge_class_list['Merge_source_dataset_class_list'].extend(
                source_dataset_class)
            self.temp_merge_class_list['Merge_target_dataset_class_list'].extend(
                target_dataset_class)

        print('Dataset instance initialize end.')
        return True

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

    def init_success(self):
        """[数据集实例初始化输出]
        """

        pass

    def get_source_dataset_image_count(self) -> int:
        """[获取源数据集图片数量]

        Returns:
            int: [源数据集图片数量]
        """

        image_count = 0
        for root, _, files in os.walk(self.dataset_input_folder):
            for n in files:
                if os.path.splitext(n)[-1].replace('.', '') in \
                        self.source_dataset_image_form_list:
                    image_count += 1

        return image_count

    def get_source_dataset_annotation_count(self) -> int:
        """[获取源数据集标注文件数量]

        Returns:
            int: [源数据集标注文件数量]
        """

        annotation_count = 0
        for root, _, files in os.walk(self.dataset_input_folder):
            for n in files:
                if n.endswith(self.source_dataset_annotation_form):
                    annotation_count += 1

        return annotation_count

    def output_classname_file(self) -> None:
        """[输出类别文件]
        """

        print('\nOutput task class name file.')
        for task, task_class_dict in self.task_dict.items():
            with open(os.path.join(self.temp_informations_folder, task + '_classes.names'), 'w') as f:
                if len(task_class_dict['Target_dataset_class']):
                    f.write('\n'.join(str(n)
                            for n in task_class_dict['Target_dataset_class']))
                f.close()

        return

    def delete_redundant_image(self) -> None:
        """[删除无标注图片]
        """

        print('\nStar delete redundant image:')
        delete_count = 0
        for n in os.listdir(self.temp_images_folder):
            image_name = os.path.splitext(n)[0]
            if image_name not in self.temp_annotation_name_list:
                delete_image_path = os.path.join(self.temp_images_folder, n)
                print('Delete redundant image: \t{}'.format(n))
                os.remove(delete_image_path)
                delete_count += 1
        print('Total delete redundant images count: {}'.format(delete_count))

        return

    def get_dataset_information(self) -> None:
        """[数据集信息分析]

        Args:
            dataset (dict): [数据集信息字典]
        """

        print('\nStart get temp dataset information:')
        self.divide_dataset()
        if self.target_dataset_style == 'cityscapes_val':
            self.image_mean_std()
            return
        self.sample_statistics()
        self.image_mean_std()
        # image_resolution_analysis(dataset)

        return

    def get_temp_annotations_name_list(self) -> list:
        """[获取暂存数据集文件名称列表]

        Args:
            dataset (dict): [数据集信息字典]

        Returns:
            list: [暂存数据集文件名称列表]
        """

        temp_file_name_list = []    # 暂存数据集文件名称列表
        print('Get temp file name list:')
        for n in tqdm(os.listdir(self.temp_annotations_folder)):
            temp_file_name_list.append(
                os.path.splitext(n.split(os.sep)[-1])[0])

        return temp_file_name_list

    def divide_dataset(self) -> None:
        """[按不同场景划分数据集，并根据不同场景按比例抽取train、val、test、redundancy比例为
        train_ratio，val_ratio，test_ratio，redund_ratio]

        Args:
            dataset (dict): [数据集信息字典]
        """

        print('\nStart divide dataset:')
        Main_path = check_output_path(self.temp_informations_folder, 'Main')
        # 统计数据集不同场景图片数量
        scene_count_dict = {}   # 场景图片计数字典
        train_dict = {}  # 训练集图片字典
        test_dict = {}  # 测试集图片字典
        val_dict = {}   # 验证集图片字典
        redund_dict = {}    # 冗余图片字典
        set_dict_list = [train_dict, val_dict, test_dict,
                         redund_dict]                                              # 数据集字典列表
        total_list = []  # 全图片列表
        # 获取全图片列表
        for one_image_name in self.temp_annotation_name_list:
            one = str(one_image_name).replace('\n', '')
            total_list.append(one)
        # 依据数据集场景划分数据集
        for image_name in total_list:                                               # 遍历全部的图片名称
            image_name_list = image_name.split(
                self.file_prefix_delimiter)                                     # 对图片名称按前缀分段，区分场景
            image_name_str = ''
            # 读取切分图片名称的值，去掉编号及后缀
            for a in image_name_list[:-1]:
                # name_str为图片包含场景的名称
                image_name_str += a
            if image_name_str in scene_count_dict.keys():                           # 判断是否已经存入场景计数字典
                # 若已经存在，则计数加1
                scene_count_dict[image_name_str][0] += 1
                scene_count_dict[image_name_str][1].append(
                    image_name)                                                     # 同时将图片名称存入对应场景分类键下
            else:
                scene_count_dict.setdefault(
                    (image_name_str), []).append(1)                                 # 若为新场景，则添加场景
                scene_count_dict[image_name_str].append(
                    [image_name])                                                   # 同时将图片名称存入对应场景分类键下
        # 计算不同场景按数据集划分比例选取样本数量
        # 遍历场景图片计数字典，获取键（不同场景）和键值（图片数、图片名称）
        if self.target_dataset_style == 'cityscapes_val':
            self.target_dataset_divide_proportion = (0, 1, 0, 0)
        for key, val in scene_count_dict.items():
            # 打包配对不同set对应不同的比例
            for diff_set_dict, diff_ratio in zip(set_dict_list, self.target_dataset_divide_proportion):
                if diff_ratio == 0:                                                 # 判断对应数据集下是否存在数据，若不存在则继续下一数据集数据挑选
                    continue
                diff_set_dict[key] = math.floor(
                    diff_ratio * val[0])                                            # 计算不同场景下不同的set应该收录的图片数
                # 依据获取的不同场景的图片数，顺序获取该数量的图片名字列表
                for a in range(diff_set_dict[key]):
                    diff_set_dict.setdefault('image_name_list', []).append(
                        scene_count_dict[key][1].pop())
        # 对分配的数据集图片名称，进行输出，分别输出为训练、测试、验证集的xml格式的txt文件
        set_name_list = ['train', 'val', 'test', 'redund']
        num_count = 0   # 图片计数
        trainval_list = []  # 训练集、验证集列表
        for set_name, set_one_path in zip(set_name_list, set_dict_list):
            print('Output images path {}.txt.'.format(set_name))
            with open(os.path.join(self.temp_informations_folder, '%s.txt' % set_name), 'w') as f:
                # 判断读取列表是否不存在，入若不存在则遍历下一数据集图片
                if len(set_one_path):
                    if self.target_dataset_style != 'cityscapes_val':
                        random.shuffle(set_one_path['image_name_list'])
                    for n in set_one_path['image_name_list']:
                        image_path = os.path.join(
                            self.temp_images_folder, n + '.' + self.target_dataset_image_form)
                        f.write('%s\n' % image_path)
                        num_count += 1
                    f.close()
                else:
                    print('No file divide to {}.'.format(set_name))
                    f.close()
                    continue
            print('Output file name {}.txt.'.format(set_name))
            with open(os.path.join(Main_path, '%s.txt' % set_name), 'w') as f:
                # 判断读取列表是否不存在，入若不存在则遍历下一数据集图片
                if len(set_one_path):
                    if self.target_dataset_style != 'cityscapes_val':
                        random.shuffle(set_one_path['image_name_list'])
                    for n in set_one_path['image_name_list']:
                        file_name = n.split(os.sep)[-1]
                        f.write('%s\n' % file_name)
                        if set_name == 'train' or set_name == 'val':
                            trainval_list.append(file_name)
                    f.close()
                else:
                    f.close()
                    continue
        print('Output file name trainval.txt.')
        with open(os.path.join(Main_path, 'trainval.txt'), 'w') as f:
            if len(trainval_list):
                f.write('\n'.join(str(n) for n in trainval_list))
                f.close()
            else:
                f.close()
        print('Output total.txt.')
        with open(os.path.join(self.temp_informations_folder, 'total.txt'), 'w') as f:
            if len(trainval_list):
                for n in total_list:
                    image_path = os.path.join(
                        self.temp_images_folder, n + '.' + self.target_dataset_image_form)
                    f.write('%s\n' % image_path)
                f.close()
            else:
                f.close()
        print('Output total_file_name.txt.')
        with open(os.path.join(self.temp_informations_folder, 'total_file_name.txt'), 'w') as f:
            if len(total_list):
                for n in total_list:
                    f.write('%s\n' % n)
                f.close()
            else:
                f.close()
        print('Total images: %d' % num_count)
        print('Divide files has been create in:\n%s' %
              self.temp_informations_folder)
        print('Divide dataset end.')

        return

    def sample_statistics(self) -> None:
        """[数据集样本统计]
        """

        # 分割后各数据集annotation文件路径
        for n in self.temp_divide_file_list:
            with open(n, 'r') as f:
                annotation_path_list = []
                for m in f.read().splitlines():
                    file_name = os.path.splitext(m.split(os.sep)[-1])[0]
                    annotation_path = os.path.join(self.temp_annotations_folder,
                                                   file_name + '.' + self.temp_annotation_form)
                    annotation_path_list.append(annotation_path)
            self.temp_divide_file_annotation_path.append(annotation_path_list)

        print('\nStar statistic sample each dataset:')
        for task, task_class_dict in self.task_dict.items():
            if task == 'Detection':
                self.detection_sample_statistics(task, task_class_dict)
            elif task == 'Semantic_segmentation':
                self.segmentation_sample_statistics(task, task_class_dict)
            elif task == 'Instance_segmentation':
                self.detection_sample_statistics(task, task_class_dict)
                self.segmentation_sample_statistics(task, task_class_dict)
            elif task == 'Keypoint':
                self.keypoint_sample_statistics(task, task_class_dict)

        return

    def detection_sample_statistics(self, task: str, task_class_dict: dict) -> None:
        """[数据集样本统计]

        Args:
            task (str): [任务类型]
            task_class_dict (dict): [任务类别字典]
        """

        # 分割后各数据集annotation文件路径
        print('Start statistic detection sample:')
        total_annotation_count_name = 'Detection_total_annotation_count.txt'
        divide_file_annotation_path = []
        for temp_annotation_path in self.temp_divide_file_list:
            with open(temp_annotation_path, 'r') as f:
                annotation_path_list = []
                for m in f.read().splitlines():
                    file_name = os.path.splitext(m.split(os.sep)[-1])[0]
                    annotation_path = os.path.join(self.temp_annotations_folder,
                                                   file_name + '.' + self.temp_annotation_form)
                    annotation_path_list.append(annotation_path)
            divide_file_annotation_path.append(annotation_path_list)

        # 声明set类别计数字典列表顺序为ttvt
        self.temp_divide_count_dict_list_dict.update({task: []})
        # 声明set类别计数字典列表顺序为ttvt
        self.temp_divide_proportion_dict_list_dict.update({task: []})
        print('\nStar statistic sample each dataset:')
        for divide_annotation_list, divide_distribution_file in \
                tqdm(zip(divide_file_annotation_path, self.temp_set_name_list),
                     total=len(divide_file_annotation_path), desc='Statistic detection sample'):
            # 声明不同集的类别计数字典
            one_set_class_count_dict = {}
            # 声明不同集的类别占比字典
            one_set_class_prop_dict = {}
            for one_class in task_class_dict['Target_dataset_class']:
                # 读取不同类别进计数字典作为键
                one_set_class_count_dict[one_class] = 0
                # 读取不同类别进占比字典作为键
                one_set_class_prop_dict[one_class] = float(0)

            # 统计全部labels各类别数量
            pbar, update = multiprocessing_list_tqdm(divide_annotation_list,
                                                     desc='Count {} class box'.format(
                                                         divide_distribution_file),
                                                     leave=False)
            process_output = multiprocessing.Manager().dict()
            pool = multiprocessing.Pool(self.workers)
            process_total_annotation_detect_class_count_dict = multiprocessing.Manager(
            ).dict({x: 0 for x in task_class_dict['Target_dataset_class']})
            for temp_annotation_path in tqdm(divide_annotation_list):
                pool.apply_async(func=self.get_temp_annotations_classes_count, args=(
                    temp_annotation_path, process_output, process_total_annotation_detect_class_count_dict,
                    task, task_class_dict,),
                    callback=update,
                    error_callback=err_call_back)
            pool.close()
            pool.join()
            pbar.close()
            for key in one_set_class_count_dict.keys():
                if key in process_output:
                    one_set_class_count_dict[key] = process_output[key]
            self.temp_divide_count_dict_list_dict[task].append(
                one_set_class_count_dict)

            # 声明单数据集计数总数
            one_set_total_count = 0
            for _, value in one_set_class_count_dict.items():    # 计算数据集计数总数
                one_set_total_count += value
            for key, value in one_set_class_count_dict.items():
                if 0 == one_set_total_count:
                    one_set_class_prop_dict[key] = 0
                else:
                    one_set_class_prop_dict[key] = (
                        float(value) / float(one_set_total_count)) * 100   # 计算个类别在此数据集占比
            self.temp_divide_proportion_dict_list_dict[task].append(
                one_set_class_prop_dict)

            # 记录每个集的类别分布
            with open(os.path.join(self.temp_sample_statistics_folder,
                                   'Detection_' + divide_distribution_file), 'w') as dist_txt:
                print('\n%s set class count:' %
                      divide_distribution_file.split('_')[0])
                for key, value in one_set_class_count_dict.items():
                    dist_txt.write(str(key) + ':' + str(value) + '\n')
                    print(str(key) + ':' + str(value))
                print('%s set porportion:' %
                      divide_distribution_file.split('_')[0])
                dist_txt.write('\n')
                for key, value in one_set_class_prop_dict.items():
                    dist_txt.write(str(key) + ':' +
                                   str('%0.2f%%' % value) + '\n')
                    print(str(key) + ':' + str('%0.2f%%' % value))

            # 记录统计标注数量
            if divide_distribution_file == 'total_distibution.txt':
                with open(os.path.join(self.temp_sample_statistics_folder,
                                       total_annotation_count_name), 'w') as dist_txt:
                    print('\n%s set class count:' %
                          divide_distribution_file.split('_')[0])
                    for key, value in one_set_class_count_dict.items():
                        dist_txt.write(str(key) + ':' + str(value) + '\n')
                        print(str(key) + ':' + str(value))
                    print('%s set porportion:' %
                          divide_distribution_file.split('_')[0])
                    dist_txt.write('\n')
                    for key, value in one_set_class_prop_dict.items():
                        dist_txt.write(str(key) + ':' +
                                       str('%0.2f%%' % value) + '\n')
                        print(str(key) + ':' + str('%0.2f%%' % value))

        self.plot_detection_sample_statistics(task, task_class_dict)    # 绘图

        return

    def segmentation_sample_statistics(self, task: str, task_class_dict: dict) -> None:
        """[语义分割样本统计]

        Args:
            task (str): [任务类型]
            task_class_dict (dict): [对应任务类别字典]
        """

        print('Start statistic semantic segmentation sample:')
        total_annotation_count_name = 'Semantic_segmentation_total_annotation_count.txt'
        # 声明set类别计数字典列表顺序为ttvt
        self.temp_divide_count_dict_list_dict.update({task: []})
        # 声明set类别计数字典列表顺序为ttvt
        self.temp_divide_proportion_dict_list_dict.update({task: []})

        for divide_annotation_list, divide_distribution_file in tqdm(zip(self.temp_divide_file_annotation_path,
                                                                         self.temp_set_name_list),
                                                                     total=len(
                                                                         self.temp_divide_file_annotation_path),
                                                                     desc='Statistic semantic segmentation sample'):
            # 声明不同集的类别计数字典
            one_set_class_pixal_dict = {}
            # 声明不同集的类别占比字典
            one_set_class_prop_dict = {}
            # 全部标注统计
            total_annotation_class_count_dict = {}
            total_annotation_class_pixal_dict = {}
            total_annotation_class_prop_dict = {}

            # 声明单数据集像素点计数总数
            one_set_total_count = 0
            for one_class in task_class_dict['Target_dataset_class']:
                # 读取不同类别进计数字典作为键
                one_set_class_pixal_dict[one_class] = 0
                # 读取不同类别进占比字典作为键
                one_set_class_prop_dict[one_class] = float(0)
                if divide_distribution_file == 'total_distibution.txt':
                    total_annotation_class_count_dict[one_class] = 0
                    total_annotation_class_prop_dict[one_class] = float(0)
            if 'unlabeled' not in one_set_class_pixal_dict:
                one_set_class_pixal_dict.update({'unlabeled': 0})
            if 'unlabeled' not in one_set_class_prop_dict:
                one_set_class_prop_dict.update({'unlabeled': 0})

            # 统计全部labels各类别像素点数量

            # TODO segmentation_sample_statistics多进程改进
            image_class_pixal_dict_list = []
            total_image_class_pixal_dict_list = []
            total_annotation_class_count_dict_list = []
            pbar, update = multiprocessing_list_tqdm(divide_annotation_list,
                                                     desc='Count {} class pixal'.format(
                                                         divide_distribution_file),
                                                     leave=False)
            pool = multiprocessing.Pool(self.workers)
            for temp_annotation_path in divide_annotation_list:
                image_class_pixal_dict_list = pool.apply_async(func=self.get_temp_segmentation_class_pixal, args=(
                    temp_annotation_path,
                    divide_distribution_file,
                    total_annotation_class_count_dict,),
                    callback=update,
                    error_callback=err_call_back)
                total_image_class_pixal_dict_list.extend(
                    image_class_pixal_dict_list.get()[0])
                total_annotation_class_count_dict_list.extend(
                    image_class_pixal_dict_list.get()[1])

            pool.close()
            pool.join()
            pbar.close()

            # 获取多进程结果
            for n in total_image_class_pixal_dict_list:
                for key, value in n.items():
                    if key in one_set_class_pixal_dict:
                        one_set_class_pixal_dict[key] += value
                    else:
                        one_set_class_pixal_dict.update({key: value})
            self.temp_divide_count_dict_list_dict[task].append(
                one_set_class_pixal_dict)

            # 计算数据集计数总数
            for _, value in one_set_class_pixal_dict.items():
                one_set_total_count += value
            for key, value in one_set_class_pixal_dict.items():
                if 0 == one_set_total_count:
                    one_set_class_prop_dict[key] = 0
                else:
                    one_set_class_prop_dict[key] = (
                        float(value) / float(one_set_total_count)) * 100  # 计算个类别在此数据集占比
            self.temp_divide_proportion_dict_list_dict[task].append(
                one_set_class_prop_dict)

            # 统计标注数量
            if divide_distribution_file == 'total_distibution.txt':
                total_annotation_count = 0
                for _, value in total_annotation_class_count_dict.items():  # 计算数据集计数总数
                    total_annotation_count += value
                for key, value in total_annotation_class_count_dict.items():
                    if 0 == total_annotation_count:
                        total_annotation_class_prop_dict[key] = 0
                    else:
                        total_annotation_class_prop_dict[key] = (
                            float(value) / float(total_annotation_count)) * 100  # 计算个类别在此数据集占比
                total_annotation_class_count_dict.update(
                    {'total': total_annotation_count})

            # 记录每个集的类别分布
            with open(os.path.join(self.temp_sample_statistics_folder,
                                   'Semantic_segmentation_' + divide_distribution_file), 'w') as dist_txt:
                print('\n%s set class pixal count:' %
                      divide_distribution_file.split('_')[0])
                for key, value in one_set_class_pixal_dict.items():
                    dist_txt.write(str(key) + ':' + str(value) + '\n')
                    print(str(key) + ':' + str(value))
                print('%s set porportion:' %
                      divide_distribution_file.split('_')[0])
                dist_txt.write('\n')
                for key, value in one_set_class_prop_dict.items():
                    dist_txt.write(str(key) + ':' +
                                   str('%0.2f%%' % value) + '\n')
                    print(str(key) + ':' + str('%0.2f%%' % value))

            # 记录统计标注数量
            if divide_distribution_file == 'total_distibution.txt':
                with open(os.path.join(self.temp_sample_statistics_folder,
                                       total_annotation_count_name), 'w') as dist_txt:
                    print('\n%s set class pixal count:' %
                          total_annotation_count_name.split('_')[0])
                    for key, value in total_annotation_class_count_dict.items():
                        dist_txt.write(str(key) + ':' + str(value) + '\n')
                        print(str(key) + ':' + str(value))
                    print('%s set porportion:' %
                          divide_distribution_file.split('_')[0])
                    dist_txt.write('\n')
                    for key, value in total_annotation_class_prop_dict.items():
                        dist_txt.write(str(key) + ':' +
                                       str('%0.2f%%' % value) + '\n')
                        print(str(key) + ':' + str('%0.2f%%' % value))

        self.plot_segmentation_sample_statistics(task, task_class_dict)    # 绘图

        # old
        #     for n in tqdm(divide_annotation_list,
        #                   desc='Count {} class pixal'.format(
        #                       divide_distribution_file),
        #                   leave=False):
        #         image = self.TEMP_LOAD(self, n)
        #         image_pixal = image.height*image.width
        #         if image == None:
        #             print('Load erro: ', n)
        #             continue
        #         for object in image.object_list:
        #             area = polygon_area(object.segmentation[:-1])
        #             if object.segmentation_clss != 'unlabeled':
        #                 one_set_class_pixal_dict[object.segmentation_clss] += area
        #                 if divide_distribution_file == 'total_distibution.txt':
        #                     total_annotation_class_count_dict[object.segmentation_clss] += 1
        #             else:
        #                 image_pixal -= area
        #                 if divide_distribution_file == 'total_distibution.txt' and \
        #                         'unlabeled' in total_annotation_class_count_dict:
        #                     total_annotation_class_count_dict[object.segmentation_clss] += 1
        #         one_set_class_pixal_dict['unlabeled'] += image_pixal

        #     self.temp_divide_count_dict_list_dict[task].append(
        #         one_set_class_pixal_dict)

        #     # 计算数据集计数总数
        #     for _, value in one_set_class_pixal_dict.items():
        #         one_set_total_count += value
        #     for key, value in one_set_class_pixal_dict.items():
        #         if 0 == one_set_total_count:
        #             one_set_class_prop_dict[key] = 0
        #         else:
        #             one_set_class_prop_dict[key] = (
        #                 float(value) / float(one_set_total_count)) * 100  # 计算个类别在此数据集占比
        #     self.temp_divide_proportion_dict_list_dict[task].append(
        #         one_set_class_prop_dict)

        #     # 统计标注数量
        #     if divide_distribution_file == 'total_distibution.txt':
        #         total_annotation_count = 0
        #         for _, value in total_annotation_class_count_dict.items():  # 计算数据集计数总数
        #             total_annotation_count += value
        #         for key, value in total_annotation_class_count_dict.items():
        #             if 0 == total_annotation_count:
        #                 total_annotation_class_prop_dict[key] = 0
        #             else:
        #                 total_annotation_class_prop_dict[key] = (
        #                     float(value) / float(total_annotation_count)) * 100  # 计算个类别在此数据集占比
        #         total_annotation_class_count_dict.update(
        #             {'total': total_annotation_count})

        #     # 记录每个集的类别分布
        #     with open(os.path.join(self.temp_sample_statistics_folder,
        #                            'Semantic_segmentation_' + divide_distribution_file), 'w') as dist_txt:
        #         print('\n%s set class pixal count:' %
        #               divide_distribution_file.split('_')[0])
        #         for key, value in one_set_class_pixal_dict.items():
        #             dist_txt.write(str(key) + ':' + str(value) + '\n')
        #             print(str(key) + ':' + str(value))
        #         print('%s set porportion:' %
        #               divide_distribution_file.split('_')[0])
        #         dist_txt.write('\n')
        #         for key, value in one_set_class_prop_dict.items():
        #             dist_txt.write(str(key) + ':' +
        #                            str('%0.2f%%' % value) + '\n')
        #             print(str(key) + ':' + str('%0.2f%%' % value))

        #     # 记录统计标注数量
        #     if divide_distribution_file == 'total_distibution.txt':
        #         with open(os.path.join(self.temp_sample_statistics_folder,
        #                                total_annotation_count_name), 'w') as dist_txt:
        #             print('\n%s set class pixal count:' %
        #                   total_annotation_count_name.split('_')[0])
        #             for key, value in total_annotation_class_count_dict.items():
        #                 dist_txt.write(str(key) + ':' + str(value) + '\n')
        #                 print(str(key) + ':' + str(value))
        #             print('%s set porportion:' %
        #                   divide_distribution_file.split('_')[0])
        #             dist_txt.write('\n')
        #             for key, value in total_annotation_class_prop_dict.items():
        #                 dist_txt.write(str(key) + ':' +
        #                                str('%0.2f%%' % value) + '\n')
        #                 print(str(key) + ':' + str('%0.2f%%' % value))

        # self.plot_segmentation_sample_statistics(task, task_class_dict)    # 绘图

        return

    def get_temp_segmentation_class_pixal(self,
                                          temp_annotation_path,
                                          divide_distribution_file,
                                          total_annotation_class_count_dict):

        image_class_pixal_dict_list = []
        total_annotation_class_count_dict_list = []

        image = self.TEMP_LOAD(self, temp_annotation_path)
        image_pixal = image.height*image.width
        if image == None:
            print('Load erro: ', temp_annotation_path)
            return
        for object in image.object_list:
            area = polygon_area(object.segmentation[:-1])
            if object.segmentation_clss != 'unlabeled':
                image_class_pixal_dict_list.append(
                    {object.segmentation_clss: area})
                if divide_distribution_file == 'total_distibution.txt':
                    total_annotation_class_count_dict_list.append(
                        {object.segmentation_clss: 1})
            else:
                image_pixal -= area
                if divide_distribution_file == 'total_distibution.txt' and \
                        'unlabeled' in total_annotation_class_count_dict:
                    total_annotation_class_count_dict_list.append(
                        {object.segmentation_clss: 1})
        image_class_pixal_dict_list.append({'unlabeled': image_pixal})

        return [image_class_pixal_dict_list, total_annotation_class_count_dict_list]

    def keypoint_sample_statistics(self, task, task_class_dict):
        """[数据集样本统计]

        Args:
            dataset (dict): [数据集信息字典]
        """

        # 分割后各数据集annotation文件路径
        total_annotation_keypoint_count_name = 'keypoint_total_annotation_count.txt'
        divide_file_annotation_path = []
        for n in self.temp_divide_file_list:
            with open(n, 'r') as f:
                annotation_path_list = []
                for m in f.read().splitlines():
                    file_name = os.path.splitext(m.split(os.sep)[-1])[0]
                    annotation_path = os.path.join(self.temp_annotations_folder,
                                                   file_name + '.' + self.temp_annotation_form)
                    annotation_path_list.append(annotation_path)
            divide_file_annotation_path.append(annotation_path_list)

        # 声明set类别计数字典列表顺序为ttvt
        self.temp_divide_count_dict_list_dict.update({task: []})
        # 声明set类别计数字典列表顺序为ttvt
        self.temp_divide_proportion_dict_list_dict.update({task: []})
        print('\nStart to statistic sample each dataset:')
        for divide_annotation_list, divide_distribution_file in \
                tqdm(zip(divide_file_annotation_path, self.temp_set_name_list),
                     total=len(divide_file_annotation_path)):
            # 声明不同集的类别计数字典
            one_set_class_count_dict = {}
            # 声明不同集的类别占比字典
            one_set_class_prop_dict = {}
            for one_class in task_class_dict['Target_dataset_class']:
                # 读取不同类别进计数字典作为键
                one_set_class_count_dict[one_class] = 0
                # 读取不同类别进占比字典作为键
                one_set_class_prop_dict[one_class] = float(0)

            # 统计全部labels各类别数量
            process_output = multiprocessing.Manager().dict()
            pool = multiprocessing.Pool(self.workers)
            process_total_annotation_detect_class_count_dict = multiprocessing.Manager(
            ).dict({x: 0 for x in task_class_dict['Target_dataset_class']})
            for n in tqdm(divide_annotation_list):
                pool.apply_async(func=self.get_temp_annotations_classes_count, args=(
                    n, process_output, process_total_annotation_detect_class_count_dict,
                    task, task_class_dict,),
                    error_callback=err_call_back)
            pool.close()
            pool.join()
            for key in one_set_class_count_dict.keys():
                if key in process_output:
                    one_set_class_count_dict[key] = process_output[key]
            self.temp_divide_count_dict_list_dict[task].append(
                one_set_class_count_dict)

            # 声明单数据集计数总数
            one_set_total_count = 0
            for _, value in one_set_class_count_dict.items():    # 计算数据集计数总数
                one_set_total_count += value
            for key, value in one_set_class_count_dict.items():
                if 0 == one_set_total_count:
                    one_set_class_prop_dict[key] = 0
                else:
                    one_set_class_prop_dict[key] = (
                        float(value) / float(one_set_total_count)) * 100   # 计算个类别在此数据集占比
            self.temp_divide_proportion_dict_list_dict[task].append(
                one_set_class_prop_dict)

            # 记录每个集的类别分布
            with open(os.path.join(self.temp_informations_folder,
                                   divide_distribution_file), 'w') as dist_txt:
                print('\n%s set class count:' %
                      divide_distribution_file.split('_')[0])
                for key, value in one_set_class_count_dict.items():
                    dist_txt.write(str(key) + ':' + str(value) + '\n')
                    print(str(key) + ':' + str(value))
                print('\n%s set porportion:' %
                      divide_distribution_file.split('_')[0])
                dist_txt.write('\n')
                for key, value in one_set_class_prop_dict.items():
                    dist_txt.write(str(key) + ':' +
                                   str('%0.2f%%' % value) + '\n')
                    print(str(key) + ':' + str('%0.2f%%' % value))

        self.plot_segmentation_sample_statistics(task, task_class_dict)    # 绘图

        return

    def get_image_mean_std(self, img_filename: str) -> list:
        """[获取图片均值和标准差]

        Args:
            dataset (dict): [数据集信息字典]
            img_filename (str): [图片名]

        Returns:
            list: [图片均值和标准差列表]
        """

        try:
            img = Image.open(os.path.join(
                self.source_dataset_images_folder, img_filename))
        except:
            print(img_filename)
            return

        img = cv2.cvtColor(np.asarray(
            img.getdata(), dtype='uint8'), cv2.COLOR_RGB2BGR)
        m, s = cv2.meanStdDev(img / 255.0)
        name = img_filename

        return m.reshape((3,)), s.reshape((3,)), name

    def image_mean_std(self) -> None:
        """[计算读取的数据集图片均值、标准差]

        Args:
            dataset (dict): [数据集信息字典]
        """

        img_filenames = os.listdir(self.source_dataset_images_folder)
        print('Start count images mean and std:')
        pbar, update = multiprocessing_list_tqdm(
            img_filenames, desc='Count images mean and std')
        pool = multiprocessing.Pool(self.workers)
        mean_std_list = []
        for img_filename in img_filenames:
            mean_std_list.append(pool.apply_async(func=self.get_image_mean_std,
                                                  args=(img_filename,),
                                                  callback=update,
                                                  error_callback=err_call_back))
        pool.close()
        pool.join()
        pbar.close()

        m_list, s_list = [], []
        for n in mean_std_list:
            m_list.append(n.get()[0])
            s_list.append(n.get()[1])
        m_array = np.array(m_list)
        s_array = np.array(s_list)
        m = m_array.mean(axis=0, keepdims=True)
        s = s_array.mean(axis=0, keepdims=True)

        mean_std_file_output_path = os.path.join(
            self.temp_informations_folder, 'mean_std.txt')
        with open(mean_std_file_output_path, 'w') as f:
            f.write('mean: ' + str(m[0][::-1]) + '\n')
            f.write('std: ' + str(s[0][::-1]))
            f.close()
        print('mean: {}'.format(m[0][::-1]))
        print('std: {}'.format(s[0][::-1]))
        print('Count images mean and std end.')

        return

    def plot_detection_sample_statistics(self, task, task_class_dict) -> None:
        """[绘制样本统计图]

        Args:
            dataset ([数据集类]): [数据集类实例]
        """

        x = np.arange(len(task_class_dict['Target_dataset_class']))  # x为类别数量
        fig = plt.figure(1, figsize=(
            len(task_class_dict['Target_dataset_class']), 9))   # 图片宽比例为类别个数

        # 绘图
        # 绘制真实框数量柱状图
        ax = fig.add_subplot(211)   # 单图显示类别计数柱状图
        ax.set_title('Dataset distribution',
                     bbox={'facecolor': '0.8', 'pad': 2})
        # width_list = [-0.45, -0.15, 0.15, 0.45]
        width_list = [0, 0, 0, 0, 0]
        colors = ['dodgerblue', 'aquamarine',
                  'pink', 'mediumpurple', 'slategrey']
        bar_width = 0
        print('Plot bar chart:')
        for one_set_label_path_list, set_size, clrs in tqdm(zip(self.temp_divide_count_dict_list_dict[task],
                                                                width_list, colors),
                                                            total=len(self.temp_divide_count_dict_list_dict[task])):
            labels = []     # class
            values = []     # class count
            # 遍历字典分别将键名和对应的键值存入绘图标签列表、绘图y轴列表中
            # for key, value in sorted(one_set_label_path_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
            for key, value in one_set_label_path_list.items():
                labels.append(str(key))
                values.append(int(value))
                bar_width = max(bar_width, int(
                    math.log10(value) if 0 != value else 1))
            # 绘制数据集类别数量统计柱状图
            ax.bar(x + set_size, values,
                   width=1, color=clrs)
            if colors.index(clrs) == 0:
                for m, b in zip(x, values):     # 为柱状图添加标签
                    plt.text(m + set_size, b, '%.0f' %
                             b, ha='center', va='bottom', fontsize=10)
            if colors.index(clrs) == 1:
                for m, b in zip(x, values):     # 为柱状图添加标签
                    plt.text(m + set_size, b, '%.0f' %
                             b, ha='center', va='top', fontsize=10, color='r')
            plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
            plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                                top=0.8, wspace=0.3, hspace=0.2)
            plt.tight_layout()
        plt.legend(['Total', 'Train', 'val', 'test', 'redund'],
                   loc='best', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

        # 绘制占比点线图
        at = fig.add_subplot(212)   # 单图显示类别占比线条图
        at.set_title('Dataset proportion',
                     bbox={'facecolor': '0.8', 'pad': 2})
        width_list = [0, 0, 0, 0, 0]
        thread_type_list = ['*', '*--', '.-.', '+-.', '-']

        print('Plot linear graph:')
        for one_set_label_path_list, set_size, clrs, thread_type in tqdm(zip(self.temp_divide_proportion_dict_list_dict[task],
                                                                             width_list, colors, thread_type_list),
                                                                         total=len(self.temp_divide_proportion_dict_list_dict[task])):
            labels = []     # class
            values = []     # class count
            # 遍历字典分别将键名和对应的键值存入绘图标签列表、绘图y轴列表中
            # for key, value in sorted(one_set_label_path_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
            for key, value in one_set_label_path_list.items():
                labels.append(str(key))
                values.append(float(value))
            # 绘制数据集类别占比点线图状图
            at.plot(x, values, thread_type, linewidth=2, color=clrs)
            if colors.index(clrs) == 0:
                for m, b in zip(x, values):     # 为图添加标签
                    plt.text(m + set_size, b, '%.2f%%' %
                             b, ha='center', va='bottom', fontsize=10)
            if colors.index(clrs) == 1:
                for m, b in zip(x, values):     # 为图添加标签
                    plt.text(m + set_size, b, '%.2f%%' %
                             b, ha='center', va='top', fontsize=10, color='r')
            plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
            plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                                top=0.8, wspace=0.3, hspace=0.2)
            plt.tight_layout()
        plt.legend(['Total', 'Train', 'val', 'test', 'redund'],
                   loc='best', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        plt.savefig(os.path.join(self.temp_sample_statistics_folder,
                                 'Detection dataset distribution.tif'), bbox_inches='tight')
        # plt.show()
        plt.close(fig)

        return

    def plot_segmentation_sample_statistics(self, task, task_class_dict) -> None:
        """[绘制样本统计图]

        Args:
            dataset ([数据集类]): [数据集类实例]
        """

        if 'unlabeled' in task_class_dict['Target_dataset_class']:
            x = np.arange(
                len(task_class_dict['Target_dataset_class']))  # x为类别数量
        else:
            x = np.arange(
                len(task_class_dict['Target_dataset_class']) + 1)  # x为类别数量
        fig = plt.figure(1, figsize=(
            len(task_class_dict['Target_dataset_class']), 9))   # 图片宽比例为类别个数

        # 绘图
        # 绘制真实框数量柱状图
        ax = fig.add_subplot(211)   # 单图显示类别计数柱状图
        ax.set_title('Dataset distribution',
                     bbox={'facecolor': '0.8', 'pad': 2})
        # width_list = [-0.45, -0.15, 0.15, 0.45]
        width_list = [0, 0, 0, 0, 0]
        colors = ['dodgerblue', 'aquamarine',
                  'pink', 'mediumpurple', 'slategrey']
        bar_width = 0

        print('Plot bar chart.')
        for one_set_label_path_list, set_size, clrs in \
            zip(self.temp_divide_count_dict_list_dict[task],
                width_list, colors):
            labels = []     # class
            values = []     # class count
            # 遍历字典分别将键名和对应的键值存入绘图标签列表、绘图y轴列表中
            # for key, value in sorted(one_set_label_path_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
            for key, value in one_set_label_path_list.items():
                labels.append(str(key))
                values.append(int(value))
                bar_width = max(bar_width, int(
                    math.log10(value) if 0 != value else 1))
            # 绘制数据集类别数量统计柱状图
            ax.bar(x + set_size, values,
                   width=1, color=clrs)
            if colors.index(clrs) == 0:
                for m, b in zip(x, values):     # 为柱状图添加标签
                    plt.text(m + set_size, b, '%.0f' %
                             b, ha='center', va='bottom', fontsize=10)
            if colors.index(clrs) == 1:
                for m, b in zip(x, values):     # 为柱状图添加标签
                    plt.text(m + set_size, b, '%.0f' %
                             b, ha='center', va='top', fontsize=10, color='r')
            plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
            plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                                top=0.8, wspace=0.3, hspace=0.2)
            plt.tight_layout()
        plt.legend(['Total', 'Train', 'val', 'test', 'redund'],
                   loc='best', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

        # 绘制占比点线图
        at = fig.add_subplot(212)   # 单图显示类别占比线条图
        at.set_title('Dataset proportion',
                     bbox={'facecolor': '0.8', 'pad': 2})
        width_list = [0, 0, 0, 0, 0]
        thread_type_list = ['*', '*--', '.-.', '+-.', '-']

        print('Plot linear graph.')
        for one_set_label_path_list, set_size, clrs, thread_type \
            in zip(self.temp_divide_proportion_dict_list_dict[task],
                   width_list, colors, thread_type_list):
            labels = []     # class
            values = []     # class count
            # 遍历字典分别将键名和对应的键值存入绘图标签列表、绘图y轴列表中
            # for key, value in sorted(one_set_label_path_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
            for key, value in one_set_label_path_list.items():
                labels.append(str(key))
                values.append(float(value))
            # 绘制数据集类别占比点线图状图
            at.plot(x, values, thread_type, linewidth=2, color=clrs)
            if colors.index(clrs) == 0:
                for m, b in zip(x, values):     # 为图添加标签
                    plt.text(m + set_size, b, '%.2f%%' %
                             b, ha='center', va='bottom', fontsize=10)
            if colors.index(clrs) == 1:
                for m, b in zip(x, values):     # 为图添加标签
                    plt.text(m + set_size, b, '%.2f%%' %
                             b, ha='center', va='top', fontsize=10, color='r')
            plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
            plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                                top=0.8, wspace=0.3, hspace=0.2)
            plt.tight_layout()
        plt.legend(['Total', 'Train', 'val', 'test', 'redund'],
                   loc='best', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        plt.savefig(os.path.join(self.temp_sample_statistics_folder,
                                 'Semantic segmentation dataset distribution.tif'), bbox_inches='tight')
        # plt.show()
        plt.close(fig)

        return

    def plot_true_box(self, task, task_class_dict) -> None:
        """[绘制每张图片的真实框检测图]

        Args:
            dataset ([Dataset]): [Dataset类实例]
            image (IMAGE): [IMAGE类实例]
        """

        # 类别色彩
        colors = [[random.randint(0, 255) for _ in range(3)]
                  for _ in range(len(task_class_dict['Target_dataset_class']))]
        # 统计各个类别的框数
        nums = [[]
                for _ in range(len(task_class_dict['Target_dataset_class']))]
        image_count = 0
        plot_true_box_success = 0
        plot_true_box_fail = 0
        total_box = 0
        print('Output check true box annotation images:')
        for image in tqdm(self.target_dataset_check_images_list):
            image_path = os.path.join(
                self.temp_images_folder, image.image_name)
            output_image = cv2.imread(image_path)  # 读取对应标签图片
            for object in image.object_list:  # 获取每张图片的bbox信息
                if not len(object.box_xywh):
                    continue
                try:
                    nums[task_class_dict['Target_dataset_class'].index(
                        object.box_clss)].append(object.box_clss)
                    color = colors[task_class_dict['Target_dataset_class'].index(
                        object.box_clss)]
                    # if dataset['target_annotation_check_mask'] == False:
                    cv2.rectangle(output_image,
                                  (int(object.box_xywh[0]),
                                   int(object.box_xywh[1])),
                                  (int(object.box_xywh[0]+object.box_xywh[2]),
                                   int(object.box_xywh[1]+object.box_xywh[3])), color, thickness=2)
                    plot_true_box_success += 1
                    # 绘制透明锚框
                    # else:
                    #     zeros1 = np.zeros((output_image.shape), dtype=np.uint8)
                    #     zeros1_mask = cv2.rectangle(zeros1, (box.xmin, box.ymin),
                    #                                 (box.xmax, box.ymax),
                    #                                 color, thickness=-1)
                    #     alpha = 1   # alpha 为第一张图片的透明度
                    #     beta = 0.5  # beta 为第二张图片的透明度
                    #     gamma = 0
                    #     # cv2.addWeighted 将原始图片与 mask 融合
                    #     mask_img = cv2.addWeighted(
                    #         output_image, alpha, zeros1_mask, beta, gamma)
                    #     output_image = mask_img
                    #     plot_true_box_success += 1
                    cv2.putText(output_image, object.box_clss, (int(object.box_xywh[0]), int(object.box_xywh[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
                except:
                    print(image.image_name + ' ' +
                          str(object.box_clss) + " is not in {} class list".format(task))
                    plot_true_box_fail += 1
                    continue
                total_box += 1
                # 输出图片
            path = os.path.join(
                self.target_dataset_annotation_check_output_folder, image.image_name)
            cv2.imwrite(path, output_image)
            image_count += 1

        # 输出检查统计
        print("\nTotal check annotations count: \t%d" % image_count)
        print('Check annotation true box count:')
        print("Plot true box success image: \t%d" % plot_true_box_success)
        print("Plot true box fail image:    \t%d" % plot_true_box_fail)
        print('True box class count:')
        for i in nums:
            if len(i) != 0:
                print(i[0] + ':' + str(len(i)))

        with open(os.path.join(self.target_dataset_annotation_check_output_folder,
                               'detect_class_count.txt'), 'w') as f:
            for i in nums:
                if len(i) != 0:
                    temp = i[0] + ':' + str(len(i)) + '\n'
                    f.write(temp)
            f.close()

        return

    def plot_true_segmentation(self, task, task_class_dict) -> None:
        """[绘制每张图片的真实分割检测图]

        Args:
            dataset (dict): [Dataset类实例]
        """

        colors = [[random.randint(0, 255) for _ in range(3)]
                  for _ in range(len(task_class_dict['Target_dataset_class']))]   # 类别色彩
        # 统计各个类别的框数
        nums = [[]
                for _ in range(len(task_class_dict['Target_dataset_class']))]
        image_count = 0
        plot_true_box_success = 0
        plot_true_box_fail = 0
        total_box = 0
        print('Output check images:')
        for image in tqdm(self.target_dataset_check_images_list):
            if task == 'Instance_segmentation' or 2 <= len(self.task_dict):
                image_path = os.path.join(
                    self.target_dataset_annotation_check_output_folder, image.image_name)
            else:
                image_path = os.path.join(
                    self.temp_images_folder, image.image_name)
            output_image = cv2.imread(image_path)  # 读取对应标签图片
            for object in image.object_list:  # 获取每张图片的bbox信息
                if not len(object.segmentation):
                    continue
                # try:
                nums[task_class_dict['Target_dataset_class'].index(
                    object.segmentation_clss)].append(object.segmentation_clss)
                class_color = colors[task_class_dict['Target_dataset_class'].index(
                    object.segmentation_clss)]
                if self.target_dataset_annotation_check_mask == False:
                    points = np.array(object.segmentation)
                    cv2.polylines(
                        output_image, pts=[points], isClosed=True, color=class_color, thickness=2)
                    plot_true_box_success += 1
                # 绘制透明真实框
                else:
                    zeros1 = np.zeros((output_image.shape), dtype=np.uint8)
                    points = np.array(object.segmentation)
                    zeros1_mask = cv2.fillPoly(
                        zeros1, pts=[points], color=class_color)
                    alpha = 1   # alpha 为第一张图片的透明度
                    beta = 0.5  # beta 为第二张图片的透明度
                    gamma = 0
                    # cv2.addWeighted 将原始图片与 mask 融合
                    mask_img = cv2.addWeighted(
                        output_image, alpha, zeros1_mask, beta, gamma)
                    output_image = mask_img
                    plot_true_box_success += 1

                cv2.putText(output_image, object.segmentation_clss,
                            (int(object.segmentation[0][0]), int(
                                object.segmentation[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
                # except:
                #     print(image.image_name + ' ' +
                #         str(object.segmentation_clss) + " is not in {} class list".format(task))
                #     plot_true_box_fail += 1
                #     continue
                total_box += 1
                # 输出图片
            path = os.path.join(
                self.target_dataset_annotation_check_output_folder, image.image_name)
            cv2.imwrite(path, output_image)
            image_count += 1

        # 输出检查统计
        print("\nTotal check annotations count: \t%d" % image_count)
        print('Check annotation true box count:')
        print("Plot true segment success image: \t%d" % plot_true_box_success)
        print("Plot true segment fail image:    \t%d" % plot_true_box_fail)
        for i in nums:
            if len(i) != 0:
                print(i[0] + ':' + str(len(i)))

        with open(os.path.join(self.target_dataset_annotation_check_output_folder,
                               'class_count.txt'), 'w') as f:
            for i in nums:
                if len(i) != 0:
                    temp = i[0] + ':' + str(len(i)) + '\n'
                    f.write(temp)
            f.close()

        return

    def plot_segmentation_annotation(self, task: str, task_class_dict: dict,
                                     image: IMAGE, segment_annotation_output_path: str) -> None:
        """[绘制分割标签图]

        Args:
            dataset (dict): [数据集信息字典]
            image (IMAGE): [图片类实例]
            segment_annotation_output_path (str): [分割标签图输出路径]
        """

        zeros = np.zeros((image.height, image.width), dtype=np.uint8)
        if len(image.true_segmentation_list):
            for seg in image.true_segmentation_list:
                class_color = task_class_dict['Target_dataset_class'].index(
                    seg.clss)
                points = np.array(seg.segmentation)
                zeros_mask = cv2.fillPoly(
                    zeros, pts=[points], color=class_color)
                cv2.imwrite(segment_annotation_output_path, zeros_mask)
        else:
            cv2.imwrite(segment_annotation_output_path, zeros)

        return

    @ staticmethod
    def TEMP_LOAD(dataset_instance, temp_annotation_path: str) -> IMAGE:
        """[读取暂存annotation]

        Args:
            dataset (dict): [数据集信息字典]
            temp_annotation_path (str): [annotation路径]

        Returns:
            IMAGE: [输出IMAGE类变量]
        """

        with open(temp_annotation_path, 'r') as f:
            data = json.loads(f.read())
            image_name = temp_annotation_path.split(
                os.sep)[-1].replace('.json', '.' + dataset_instance.temp_image_form)
            image_path = os.path.join(
                dataset_instance.temp_images_folder, image_name)
            if os.path.splitext(image_path)[-1] == 'png':
                img = Image.open(image_path)
                height, width = img.height, img.width
                channels = 3
            else:
                image_size = cv2.imread(image_path).shape
                height = int(image_size[0])
                width = int(image_size[1])
                channels = int(image_size[2])

            object_list = []
            for object in data['frames'][0]['objects']:
                one_object = OBJECT(object['id'],
                                    object['object_clss'],
                                    object['box_clss'],
                                    object['segmentation_clss'],
                                    object['keypoints_clss'],
                                    object['box_xywh'],
                                    object['segmentation'],
                                    object['keypoints_num'], object['keypoints'],
                                    dataset_instance.task_convert,
                                    box_color=object['box_color'],
                                    box_tool=object['box_tool'],
                                    box_difficult=object['box_difficult'],
                                    box_distance=object['box_distance'],
                                    box_occlusion=object['box_occlusion'],
                                    segmentation_area=object['segmentation_area'],
                                    segmentation_iscrowd=object['segmentation_iscrowd']
                                    )
                object_list.append(one_object)
            image = IMAGE(image_name, image_name,
                          image_path, height, width, channels, object_list)
            f.close()

        return image

    def get_temp_annotations_classes_count(self,
                                           temp_annotation_path: str,
                                           process_output: dict,
                                           process_total_annotation_detect_class_count_dict: dict,
                                           task: str, task_class_dict: dict) -> None:
        """[获取暂存标签信息]

        Args:
            dataset (dict): [数据集信息字典]
            temp_annotation_path (str): [暂存标签路径]
            process_output (dict): [进程输出字典]
        """

        image = self.TEMP_LOAD(self, temp_annotation_path)
        if task == 'Detection':
            for object in image.object_list:
                if object.box_clss in process_output:
                    process_output[object.box_clss] += 1
                    process_total_annotation_detect_class_count_dict[object.box_clss] += 1
                else:
                    process_output.update({object.box_clss: 1})
                    process_total_annotation_detect_class_count_dict.update(
                        {object.box_clss: 1})
        elif task == 'Semantic_segmentation' or task == 'Instance_segmentation':
            for object in image.object_list:
                if object.segmentation_clss in process_output:
                    process_output[object.segmentation_clss] += 1
                    process_total_annotation_detect_class_count_dict[object.segmentation_clss] += 1
                else:
                    process_output.update({object.segmentation_clss: 1})
                    process_total_annotation_detect_class_count_dict.update(
                        {object.segmentation_clss: 1})
        elif task == 'Keypoint':
            for object in image.object_list:
                if object.keypoints_clss in process_output:
                    process_output[object.keypoints_clss] += 1
                    process_total_annotation_detect_class_count_dict[object.keypoints_clss] += 1
                else:
                    process_output.update({object.keypoints_clss: 1})
                    process_total_annotation_detect_class_count_dict.update(
                        {object.keypoints_clss: 1})

        return

    def transform_to_target_dataset():
        # print('\nStart transform to target dataset:')
        raise NotImplementedError("ERROR: func not implemented!")

    def target_dataset_annotation_check(self) -> None:
        """[进行标签检测]

        Args:
            dataset (dict): [数据集信息字典]
        """

        print('\nStart check target annotations:')
        self.target_dataset_check_images_list = dataset.__dict__[
            self.target_dataset_style].annotation_check(self)
        shutil.rmtree(self.target_dataset_annotation_check_output_folder)
        check_output_path(self.target_dataset_annotation_check_output_folder)
        for task, task_class_dict in self.task_dict.items():
            if task == 'Detection':
                self.plot_true_box(task, task_class_dict)
            elif task == 'Semantic_segmentation':
                self.plot_true_segmentation(task, task_class_dict)
            elif task == 'Instance_segmentation' or \
                    task == 'Multi_task':
                self.plot_true_box(task, task_class_dict)
                self.plot_true_segmentation(task, task_class_dict)

        return

    def plot_true_segment(dataset: dict) -> None:
        """[绘制每张图片的真实分割检测图]

        Args:
            dataset (dict): [Dataset类实例]
        """

        colors = [[random.randint(0, 255) for _ in range(3)]
                  for _ in range(len(dataset['segment_class_list_new']))]   # 类别色彩
        # 统计各个类别的像素点
        nums = [[] for _ in range(len(dataset['segment_class_list_new']))]
        image_count = 0
        plot_true_box_success = 0

        print('Output check true segmentation annotation images:')
        for image in tqdm(dataset['check_images_list']):
            image_path = os.path.join(
                dataset['check_annotation_output_folder'], image.image_name)
            output_image = cv2.imread(image_path)  # 读取对应标签图片
            for object in image.true_segmentation_list:  # 获取每张图片的bbox信息
                nums[dataset['segment_class_list_new'].index(
                    object.clss)].append(object.clss)
                class_color = colors[dataset['segment_class_list_new'].index(
                    object.clss)]
                if dataset['target_segment_annotation_check_mask'] == False:
                    points = np.array(object.segmentation)
                    cv2.polylines(
                        output_image, pts=[points], isClosed=True, color=class_color, thickness=2)
                    plot_true_box_success += 1
                # 绘制透明分割真实框
                else:
                    zeros1 = np.zeros((output_image.shape), dtype=np.uint8)
                    points = np.array(object.segmentation)
                    zeros1_mask = cv2.fillPoly(
                        zeros1, pts=[points], color=class_color)
                    alpha = 1   # alpha 为第一张图片的透明度
                    beta = 0.3  # beta 为第二张图片的透明度
                    gamma = 0
                    # cv2.addWeighted 将原始图片与 mask 融合
                    mask_img = cv2.addWeighted(
                        output_image, alpha, zeros1_mask, beta, gamma)
                    output_image = mask_img
                    plot_true_box_success += 1

                cv2.putText(output_image, object.clss, (int(object.segmentation[0][0]), int(object.segmentation[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
                # 输出图片
            path = os.path.join(
                dataset['check_annotation_output_folder'], image.image_name)
            cv2.imwrite(path, output_image)
            image_count += 1

        # 输出检查统计
        print("\nTotal check annotations count: \t%d" % image_count)
        print('Check annotation true segment count:')
        print("Plot true segment success image: \t%d" % plot_true_box_success)
        print("Plot true segment fail image:    \t%d" %
              (len(dataset['check_images_list']) - image_count))
        print('True box class count:')
        for i in nums:
            if len(i) != 0:
                print(i[0] + ':' + str(len(i)))

        with open(os.path.join(dataset['check_annotation_output_folder'], 'segment_class_count.txt'), 'w') as f:
            for i in nums:
                if len(i) != 0:
                    temp = i[0] + ':' + str(len(i)) + '\n'
                    f.write(temp)
            f.close()

        return

    def plot_segment_annotation(dataset: dict, image: IMAGE, segment_annotation_output_path: str) -> None:
        """[绘制分割标签图]

        Args:
            dataset (dict): [数据集信息字典]
            image (IMAGE): [图片类实例]
            segment_annotation_output_path (str): [分割标签图输出路径]
        """

        zeros = np.zeros((image.height, image.width), dtype=np.uint8)
        if len(image.true_segmentation_list):
            for seg in image.true_segmentation_list:
                class_color = dataset['segment_class_list_new'].index(
                    seg.clss)
                points = np.array(seg.segmentation)
                zeros_mask = cv2.fillPoly(
                    zeros, pts=[points], color=class_color)
                cv2.imwrite(segment_annotation_output_path, zeros_mask)
        else:
            cv2.imwrite(segment_annotation_output_path, zeros)

        return

    def plot_pick_class_segment_annotation(dataset: dict, image: IMAGE, segment_annotation_output_path: str, class_list: list, lane_color: int) -> None:
        """[绘制分割标签图]

        Args:
            dataset (dict): [数据集信息字典]
            image (IMAGE): [图片类实例]
            segment_annotation_output_path (str): [分割标签图输出路径]
        """

        zeros = np.zeros((image.height, image.width), dtype=np.uint8)
        if len(image.true_segmentation_list):
            plot_true_segmentation_count = 0
            for seg in image.true_segmentation_list:
                if seg.clss not in class_list:
                    continue
                class_color = lane_color
                points = np.array(seg.segmentation)
                zeros_mask = cv2.fillPoly(
                    zeros, pts=[points], color=class_color)
                cv2.imwrite(segment_annotation_output_path, zeros_mask)
                plot_true_segmentation_count += 1
            if 0 == plot_true_segmentation_count:
                cv2.imwrite(segment_annotation_output_path, zeros)
        else:
            cv2.imwrite(segment_annotation_output_path, zeros)

        return

    def build_target_dataset_folder():
        # print('\nStart build target dataset folder:')
        raise NotImplementedError("ERROR: func not implemented!")
