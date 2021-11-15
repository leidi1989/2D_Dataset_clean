'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-10 18:38:55
LastEditors: Leidi
LastEditTime: 2021-11-15 15:11:45
'''
from utils.plot import plot_segment_sample_statistics, plot_detect_sample_statistics
from annotation.annotation_temp import TEMP_LOAD
from base.image_base import *
from utils.utils import *

import os
import cv2
import math
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='AR PL UMing CN')
plt.switch_backend('agg')


def information(dataset: dict) -> None:
    """[数据集信息分析]

    Args:
        dataset (dict): [数据集信息字典]
    """

    divide_dataset(dataset)
    if dataset['target_dataset_style'] == 'cityscapes_val':
        return
    detect_sample_statistics(dataset)
    segment_sample_statistics(dataset)
    image_mean_std(dataset)
    # image_resolution_analysis(dataset)

    return


def temp_file_name(dataset: dict) -> list:
    """[获取暂存数据集文件名称列表]

    Args:
        dataset (dict): [数据集信息字典]

    Returns:
        list: [暂存数据集文件名称列表]
    """

    temp_file_name_list = []    # 暂存数据集文件名称列表
    print('Get temp file name list:')
    for n in tqdm(os.listdir(dataset['temp_annotations_folder'])):
        temp_file_name_list.append(n.split(os.sep)[-1].split('.')[0])

    return temp_file_name_list


def get_TEMP_LOAD(dataset: dict, temp_annotation_path: str, process_output: dict) -> None:
    """[获取暂存标签信息]

    Args:
        dataset (dict): [数据集信息字典]
        temp_annotation_path (str): [暂存标签路径]
        process_output (dict): [进程输出字典]
    """
    image = TEMP_LOAD(dataset, temp_annotation_path)
    for m in image.true_box_list:
        if m.clss in process_output:
            process_output[m.clss] += 1
        else:
            process_output.update({m.clss: 1})

    return


def divide_dataset(dataset: dict) -> None:
    """[按不同场景划分数据集，并根据不同场景按比例抽取train、val、test、redundancy比例为
    train_ratio，val_ratio，test_ratio，redund_ratio]

    Args:
        dataset (dict): [数据集信息字典]
    """

    Main_path = check_output_path(dataset['temp_informations_folder'], 'Main')
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
    for one_image_name in dataset['temp_file_name_list']:
        one = str(one_image_name).replace('\n', '')
        total_list.append(one)
    # 依据数据集场景划分数据集
    for image_name in total_list:                                               # 遍历全部的图片名称
        image_name_list = image_name.split(
            dataset['prefix_delimiter'])                                     # 对图片名称按前缀分段，区分场景
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
    for key, val in scene_count_dict.items():
        # 打包配对不同set对应不同的比例
        for diff_set_dict, diff_ratio in zip(set_dict_list, dataset['proportion']):
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
        print('\nOutput images path {}.txt:'.format(set_name))
        with open(os.path.join(dataset['temp_informations_folder'], '%s.txt' % set_name), 'w') as f:
            # 判断读取列表是否不存在，入若不存在则遍历下一数据集图片
            if len(set_one_path):
                if dataset['source_dataset_stype'] != 'cityscapes_val':
                    random.shuffle(set_one_path['image_name_list'])
                for n in tqdm(set_one_path['image_name_list']):
                    image_path = os.path.join(
                        dataset['temp_images_folder'], n + '.' + dataset['target_image_form'])
                    f.write('%s\n' % image_path)
                    num_count += 1
                f.close()
            else:
                print('No file divide to {}.'.format(set_name))
                f.close()
                continue
        print('\nOutput file name {}.txt :'.format(set_name))
        with open(os.path.join(Main_path, '%s.txt' % set_name), 'w') as f:
            # 判断读取列表是否不存在，入若不存在则遍历下一数据集图片
            if len(set_one_path):
                if dataset['source_dataset_stype'] != 'cityscapes_val':
                    random.shuffle(set_one_path['image_name_list'])
                for n in tqdm(set_one_path['image_name_list']):
                    file_name = n.split(os.sep)[-1].split('.')[0]
                    f.write('%s\n' % file_name)
                    if set_name == 'train' or set_name == 'val':
                        trainval_list.append(file_name)
                f.close()
            else:
                f.close()
                continue
    print('\nOutput file name trainval.txt:')
    with open(os.path.join(Main_path, 'trainval.txt'), 'w') as f:
        if len(trainval_list):
            f.write('\n'.join(str(n) for n in tqdm(trainval_list)))
            f.close()
        else:
            f.close()
    print('\nOutput total.txt:')
    with open(os.path.join(dataset['temp_informations_folder'], 'total.txt'), 'w') as f:
        if len(trainval_list):
            for n in tqdm(total_list):
                image_path = os.path.join(
                    dataset['temp_images_folder'], n + '.' + dataset['target_image_form'])
                f.write('%s\n' % image_path)
            f.close()
        else:
            f.close()
    print('\nOutput total_file_name.txt:')
    with open(os.path.join(dataset['temp_informations_folder'], 'total_file_name.txt'), 'w') as f:
        if len(total_list):
            for n in tqdm(total_list):
                f.write('%s\n' % n)
            f.close()
        else:
            f.close()
    print('\nTotal images: %d' % num_count)
    print('\nDivide files has been create in %s\n' %
          dataset['temp_informations_folder'])

    return


def detect_sample_statistics(dataset: dict) -> None:
    """[数据集样本统计]

    Args:
        dataset (dict): [数据集信息字典]
    """
    # 分割后各数据集annotation文件路径
    set_name_list = ['detect_total_distibution.txt', 'detect_train_distibution.txt',
                     'detect_val_distibution.txt', 'detect_test_distibution.txt',
                     'detect_redund_distibution.txt']

    divide_file_annotation_path = []
    for n in dataset['temp_divide_file_list']:
        with open(n, 'r') as f:
            annotation_path_list = []
            for m in f.read().splitlines():
                file_name = os.path.splitext(m.split(os.sep)[-1])[0]
                annotation_path = os.path.join(dataset['temp_annotations_folder'],
                                               file_name + '.' + dataset['temp_annotation_form'])
                annotation_path_list.append(annotation_path)
        divide_file_annotation_path.append(annotation_path_list)

    # 声明set类别计数字典列表顺序为ttvt
    dataset['temp_divide_count_dict_list'] = []
    # 声明set类别计数字典列表顺序为ttvt
    dataset['temp_divide_proportion_dict_list'] = []
    for divide_annotation_list, divide_distribution_file in tqdm(zip(divide_file_annotation_path,
                                                                     set_name_list),
                                                                 total=len(divide_file_annotation_path)):
        # 声明不同集的类别计数字典
        one_set_class_count_dict = {}
        # 声明不同集的类别占比字典
        one_set_class_prop_dict = {}
        for one_class in dataset['detect_class_list_new']:
            # 读取不同类别进计数字典作为键
            one_set_class_count_dict[one_class] = 0
            # 读取不同类别进占比字典作为键
            one_set_class_prop_dict[one_class] = float(0)

        # 统计全部labels各类别数量
        with multiprocessing.Manager() as manager:
            process_output = manager.dict()
            with tqdm(total=len(divide_annotation_list)) as t:
                pool = multiprocessing.Pool(8)
                for n in divide_annotation_list:
                    pool.apply_async(func=get_TEMP_LOAD, args=(
                        dataset, n, process_output,),
                        callback=t.update(),
                        error_callback=err_call_back)
                pool.close()
                pool.join()
            for key in one_set_class_count_dict.keys():
                if key in process_output:
                    one_set_class_count_dict[key] = process_output[key]
        dataset['temp_divide_count_dict_list'].append(one_set_class_count_dict)

        # 声明单数据集计数总数
        one_set_total_count = 0
        for _, value in one_set_class_count_dict.items():                                                       # 计算数据集计数总数
            one_set_total_count += value
        for key, value in one_set_class_count_dict.items():
            if 0 == one_set_total_count:
                one_set_class_prop_dict[key] = 0
            else:
                one_set_class_prop_dict[key] = (
                    float(value) / float(one_set_total_count)) * 100                                            # 计算个类别在此数据集占比
        dataset['temp_divide_proportion_dict_list'].append(
            one_set_class_prop_dict)
        # 记录每个集的类别分布
        with open(os.path.join(dataset['temp_informations_folder'],
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

    plot_detect_sample_statistics(dataset)    # 绘图

    return


def segment_sample_statistics(dataset: dict) -> None:
    """[数据集样本统计]

    Args:
        dataset (dict): [数据集信息字典]
    """

    # 分割后各数据集annotation文件路径
    set_name_list = ['segment_total_distibution.txt', 'segment_train_distibution.txt',
                     'segment_val_distibution.txt', 'segment_test_distibution.txt',
                     'segment_redund_distibution.txt']
    divide_file_annotation_path = []
    for n in dataset['temp_divide_file_list']:
        with open(n, 'r') as f:
            annotation_path_list = []
            for m in f.read().splitlines():
                file_name = m.split(os.sep)[-1].split('.')[0]
                annotation_path = os.path.join(dataset['temp_annotations_folder'],
                                               file_name + '.' + dataset['temp_annotation_form'])
                annotation_path_list.append(annotation_path)
        divide_file_annotation_path.append(annotation_path_list)
    # 声明set类别计数字典列表顺序为ttvt
    dataset['temp_divide_count_dict_list'] = []
    # 声明set类别计数字典列表顺序为ttvt
    dataset['temp_divide_proportion_dict_list'] = []
    print('\nStart to statistic sample each dataset:')
    for divide_annotation_list, divide_distribution_file in tqdm(zip(divide_file_annotation_path,
                                                                     set_name_list),
                                                                 total=len(divide_file_annotation_path)):
        # 声明不同集的类别计数字典
        one_set_class_pixal_dict = {}
        # 声明不同集的类别占比字典
        one_set_class_prop_dict = {}
        # 声明单数据集像素点计数总数
        one_set_total_count = 0
        for one_class in dataset['segment_class_list_new']:
            # 读取不同类别进计数字典作为键
            one_set_class_pixal_dict[one_class] = 0
            # 读取不同类别进占比字典作为键
            one_set_class_prop_dict[one_class] = float(0)
        # 统计全部labels各类别像素点数量
        for n in tqdm(divide_annotation_list):
            image = TEMP_LOAD(dataset, n)
            for m in image.true_segmentation_list:
                one_set_class_pixal_dict[m.clss] += polygon_area(
                    m.segmentation[:-1])
        dataset['temp_divide_count_dict_list'].append(one_set_class_pixal_dict)
        for _, value in one_set_class_pixal_dict.items():                       # 计算数据集计数总数
            one_set_total_count += value
        for key, value in one_set_class_pixal_dict.items():
            if 0 == one_set_total_count:
                one_set_class_prop_dict[key] = 0
            else:
                one_set_class_prop_dict[key] = (
                    float(value) / float(one_set_total_count)) * 100            # 计算个类别在此数据集占比
        dataset['temp_divide_proportion_dict_list'].append(
            one_set_class_prop_dict)
        # 记录每个集的类别分布
        with open(os.path.join(dataset['temp_informations_folder'],
                               divide_distribution_file), 'w') as dist_txt:
            print('\n%s set class pixal count:' %
                  divide_distribution_file.split('_')[0])
            for key, value in one_set_class_pixal_dict.items():
                dist_txt.write(str(key) + ':' + str(value) + '\n')
                print(str(key) + ':' + str(value))
            print('\n%s set porportion:' %
                  divide_distribution_file.split('_')[0])
            dist_txt.write('\n')
            for key, value in one_set_class_prop_dict.items():
                dist_txt.write(str(key) + ':' +
                               str('%0.2f%%' % value) + '\n')
                print(str(key) + ':' + str('%0.2f%%' % value))

    plot_segment_sample_statistics(dataset)    # 绘图

    return


def get_image_mean_std(dataset: dict, img_filename: str) -> list:
    """[获取图片均值和标准差]

    Args:
        dataset (dict): [数据集信息字典]
        img_filename (str): [图片名]

    Returns:
        list: [图片均值和标准差列表]
    """

    img = cv2.imread(os.path.join(
        dataset['source_images_folder'], img_filename))
    img = img / 255.0
    m, s = cv2.meanStdDev(img)

    return m.reshape((3,)), s.reshape((3,))


def image_mean_std(dataset: dict) -> None:
    """[计算读取的数据集图片均值、标准差]

    Args:
        dataset (dict): [数据集信息字典]
    """

    img_filenames = os.listdir(dataset['source_images_folder'])
    print('Start count images mean and std:')
    pool = multiprocessing.Pool(dataset['workers'])
    mean_std_list = []
    for img_filename in tqdm(img_filenames):
        mean_std_list.append(pool.apply_async(func=get_image_mean_std, args=(
            dataset, img_filename), error_callback=err_call_back))
    pool.close()
    pool.join()

    m_list, s_list = [], []
    for n in mean_std_list:
        m_list.append(n.get()[0])
        s_list.append(n.get()[1])
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)

    mean_std_file_output_path = os.path.join(
        dataset['temp_informations_folder'], 'mean_std.txt')
    with open(mean_std_file_output_path, 'w') as f:
        f.write('mean: ' + str(m[0][::-1]) + '\n')
        f.write('std: ' + str(s[0][::-1]))
        f.close()
    print('mean: {}'.format(m[0][::-1]))
    print('std: {}'.format(s[0][::-1]))

    return


# def image_resolution_analysis(dataset: dict) -> None:
#     """[计算数据集分辨率宽高的最大值、最小值及中位数]

#     Args:
#         dataset (dict): [数据集信息字典]
#     """
#     image_path_file = os.path.join(
#         dataset['temp_informations_folder'], dataset['temp_divide_file_list'][0])
#     image_resolution_list = []
#     print('Start count dataset image resolution min, max, median:')
#     with open(image_path_file, 'r') as f:
#         for n in f.readlines():
#             image = cv2.imread(n.rstrip())
#             height, width, _ = image.shape
#             image_resolution_list.append(np.array([height, width]))
#     image_resolution_min = np.min(
#         np.array(image_resolution_list), axis=0).astype(int)
#     image_resolution_max = np.max(
#         np.array(image_resolution_list), axis=0).astype(int)
#     image_resolution_median = np.median(
#         np.array(image_resolution_list), axis=0, overwrite_input=True).astype(int)
#     dataset['image_resolution_median'] = image_resolution_median
#     print('Dataset image resolution min = {}'.format(image_resolution_min))
#     print('Dataset image resolution max = {}'.format(image_resolution_max))
#     print('Dataset image resolution median = {}'.format(image_resolution_median))

#     resolution_median_output_path = os.path.join(
#         dataset['temp_informations_folder'], 'image_resolution_median.txt')
#     print('Output dataset image resolution min, max, median file to {}'.format(
#         resolution_median_output_path))
#     with open(resolution_median_output_path, 'w') as f:
#         f.write('Dataset image resolution min = {}\n'.format(
#             image_resolution_min))
#         f.write('Dataset image resolution max = {}\n'.format(
#             image_resolution_max))
#         f.write('Dataset image resolution median = {}'.format(
#             image_resolution_median))
#         f.close()

#     return
