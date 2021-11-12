'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:02
LastEditors: Leidi
LastEditTime: 2021-10-21 20:08:58
'''
# -*- coding: utf-8 -*-
import argparse
import os
import random
import math
from tqdm import tqdm

from utils.utils import *


def divide_set(output_path, input_label_style, ratio, segment):
    """按不同场景划分数据集，并根据不同场景按比例抽取train、val、test、redundancy比例为
        train_ratio，val_ratio，test_ratio，redund_ratio"""
    
    input_path = check_input_path(output_path)
    images_path = check_output_path(os.path.join(output_path,'JPEGImages'))
    ImageSets_path = check_output_path(input_path, 'ImageSets')
    Main_path = check_output_path(ImageSets_path, 'Main')
    total_image_name = open(check_output_path(ImageSets_path, 'total_image_name.txt'))
    image_type = check_image_type(input_label_style)
    train_ratio = ratio[0]
    val_ratio = ratio[1]
    test_ratio = ratio[2]
    redund_ratio = ratio[3]
    ratio_list = [train_ratio, val_ratio, test_ratio, redund_ratio]

    # 统计数据集不同场景图片数量
    scene_count_dict = {}   # 场景图片计数字典
    train_dict = {}     # 训练集图片字典
    test_dict = {}     # 测试集图片字典
    val_dict = {}     # 验证集图片字典
    redund_dict = {}     # 冗余图片字典
    set_dict_list = [train_dict, val_dict, test_dict, redund_dict]   # 数据集字典列表
    total_list = []     # 全图片列表
    for one_image_name in total_image_name:     # 获取全图片列表
        one = str(one_image_name).replace('\n', '')
        total_list.append(one)

    # 依据数据集场景划分数据集
    for image_name in total_list:     # 遍历全部的图片名称
        image_name_list = image_name.split(segment)     # 对图片名称进行分段，区分场景
        image_name_str = ''
        # if input_label_style == 'ccpd':
        #     for a in image_name_list[0]:    # 读取切分图片名称的值，去掉编号及后缀
        #         image_name_str += a   # name_str为图片包含场景的名称
        #     if image_name_str in scene_count_dict.keys():   # 判断是否已经存入场景计数字典
        #         scene_count_dict[image_name_str][0] += 1    # 若已经存在，则计数加1
        #         scene_count_dict[image_name_str][1].append(image_name)      # 同时将图片名称存入对应场景分类键下
        #     else:
        #         scene_count_dict.setdefault((image_name_str), []).append(1)     # 若为新场景，则添加场景
        #         scene_count_dict[image_name_str].append([image_name])       # 同时将图片名称存入对应场景分类键下
        # else:
        for a in image_name_list[:-1]:    # 读取切分图片名称的值，去掉编号及后缀
            image_name_str += a   # name_str为图片包含场景的名称
        if image_name_str in scene_count_dict.keys():   # 判断是否已经存入场景计数字典
            scene_count_dict[image_name_str][0] += 1    # 若已经存在，则计数加1
            scene_count_dict[image_name_str][1].append(image_name)      # 同时将图片名称存入对应场景分类键下
        else:
            scene_count_dict.setdefault((image_name_str), []).append(1)     # 若为新场景，则添加场景
            scene_count_dict[image_name_str].append([image_name])       # 同时将图片名称存入对应场景分类键下

    # 计算不同场景按数据集划分比例选取样本数量
    for key, val in scene_count_dict.items():   # 遍历场景图片计数字典，获取键（不同场景）和键值（图片数、图片名称）
        for diff_set_dict, diff_ratio in zip(set_dict_list, ratio_list):    # 打包配对不同set对应不同的比例
            if diff_ratio == 0:     # 判断对应数据集下是否存在数据，若不存在则继续下一数据集数据挑选
                continue
            diff_set_dict[key] = math.floor(diff_ratio * val[0])        # 计算不同场景下不同的set应该收录的图片数
            for a in range(diff_set_dict[key]):     # 依据获取的不同场景的图片数，顺序获取该数量的图片名字列表
                diff_set_dict.setdefault('image_name_list', []).append(
                    scene_count_dict[key][1].pop())

    # 对分配的数据集图片名称，进行输出，分别输出为训练、测试、验证集的xml格式的txt文件
    set_name_list = ['train', 'val', 'test', 'redund']
    num_count = 0
    trainval_list = []
    for set_name, set_one_path in zip(set_name_list, set_dict_list):
        with open(os.path.join(ImageSets_path, '%s.txt' % set_name), 'w') as f:
            if len(set_one_path) == 0:   # 判断读取列表是否不存在，入若不存在则遍历下一数据集图片
                f.close()
                continue
            random.shuffle(set_one_path['image_name_list'])
            for w in tqdm(set_one_path['image_name_list']):
                x = os.path.join(images_path, w + '.' + image_type)
                f.write('%s\n' % x)
                num_count += 1
            f.close()
        with open(os.path.join(Main_path, '%s.txt' % set_name), 'w') as f:
            if len(set_one_path) == 0:   # 判断读取列表是否不存在，入若不存在则遍历下一数据集图片
                f.close()
                continue
            random.shuffle(set_one_path['image_name_list'])
            for w in tqdm(set_one_path['image_name_list']):
                f.write('%s\n' % w)
                if set_name == 'train' or set_name == 'val':
                    trainval_list.append(w)
                num_count += 1
            f.close()
            
    with open(os.path.join(Main_path, 'trainval.txt'), 'w') as f:
        for w in tqdm(trainval_list):
            f.write('%s\n' % w)

    with open(os.path.join(ImageSets_path, 'total.txt'), 'w') as f:
        for w in total_list:
            x = os.path.join(images_path, w + '.' + image_type)
            f.write('%s\n' % x)
        f.close()
    
    print('Total images: %d, 4 txt files in %s\n' % (num_count, ImageSets_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cleaning_6_divideset.py')
    parser.add_argument('--out', default=r'/home/leidi/Dataset/hy_highway_myxb_sjt_coco2017_7_classes_output_20210805',
                        type=str, help='output path')
    parser.add_argument('--segment', '--sg', dest='segment', default='_',
                        type=str, help='name split')
    parser.add_argument('--ratio', '--ra', dest='ratio', default=(0.92, 0.04, 0.04, 0),
                        nargs=3, type=float, help='train, val, test, redundancy')
    parser.add_argument('--ilstyle', '--is', dest='ilstyle', default=r'ldp',
                        type=str, help='input labels style: ldp, hy, myxb, nuscenes, \
                                                            pascal, hy_highway, coco2017, \
                                                            kitti, cctsdb, lisa, \
                                                            hanhe，yolov5_detect, yolo, \
                                                            sjt, ccpd')
    
    opt = parser.parse_args()
    output_path = check_output_path(opt.out)
    segment = opt.segment
    ratio = opt.ratio
    input_label_style = opt.ilstyle
    
    print('\nStart to divide dataset：')
    divide_set(output_path, input_label_style, ratio, segment)
    print('Divide set done!')
