'''
Description:
Version:
Author: Leidi
Date: 2021-04-26 20:59:03
LastEditors: Leidi
LastEditTime: 2021-10-21 20:09:07
'''
# -*- coding: utf-8 -*-
import os
import random
import codecs
import cv2
import csv
import json as js
import numpy as np
import operator
from tqdm import tqdm
import xml.etree.ElementTree as ET

from utils.utils import *
from utils.convertion_function import *


class true_box:
    """真实框类"""

    def __init__(self, clss, xmin, ymin, xmax, ymax, color='', tool='', difficult=0, distance=0, occlusion=0):
        self.clss = clss
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.color = color  # 真实框目标颜色
        self.tool = tool    # bbox工具
        self.difficult = difficult
        self.distance = distance    # 真实框中心点距离
        self.occlusion = occlusion    # 真实框遮挡率


class per_image:
    """图片类"""

    def __init__(self, image_name_in, image_path_in, height_in, width_in, channels_in, true_box_list_in):
        self.image_name = image_name_in    # 图片名称
        self.image_path = image_path_in    # 图片地址
        self.height = height_in    # 图片高
        self.width = width_in  # 图片宽
        self.channels = channels_in    # 图片通道数
        self.true_box_list = true_box_list_in  # 图片真实框列表
        # self.free_area = len(self.free_space_area())    # 获取图片掩码图的非真实框面积

    def true_box_list_updata(self, one_bbox_data):
        """[为per_image对象true_box_list成员添加元素]

        Parameters
        ----------
        one_bbox_data : [class true_box]
            [真实框类]
        """

        self.true_box_list.append(one_bbox_data)

    def get_true_box_mask(self):
        """[获取图片真实框掩码图，前景置1，背景置0]

        Parameters
        ----------
        true_box_list : [list]
            [真实框列表]
        """

        image_mask = np.zeros([self.height, self.width])
        for one_ture_box in self.true_box_list:     # 读取true_box并对前景在mask上置1
            image_mask[int(one_ture_box.ymin):int(one_ture_box.ymax),
                       int(one_ture_box.xmin):int(one_ture_box.xmax)] = 1     # 将真实框范围内置1

        return image_mask

    # TODO
    def free_space_area(self):
        """[获取图片非真实框像素列表]

        Returns
        -------
        free_space_area_list : [list]
            [图片非真实框像素列表]
        """

        mask_true_box = self.get_true_box_mask()

        return np.argwhere(mask_true_box == 0)


def from_ldp(input_path, class_list):
    """ldp格式，抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'filename'
    image_path = os.path.join(input_path, 'images')   # 图片路径
    image_path_voc = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(input_path, 'Annotations')  # 对应图片的json文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    if not len(src_lab_path_list):
        src_lab_path = os.path.join(
            input_path, 'source_label')  # 对应图片的json文件路径
        src_lab_path_list = os.listdir(
            src_lab_path)  # 读取source label文件夹下的全部文件名
    total_images_data_list = []     # 声明全部图片列表
    print('Start to load each annotation data file:')
    # 将每一个Annotations文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        try:
            tree = ET.parse(src_lab_dir)
            root = tree.getroot()
            image_name = str(root.find(key).text)
            img_path = os.path.join(image_path_voc, image_name)
            img = cv2.imread(img_path)
            image_size = img.shape
            height = image_size[0]
            width = image_size[1]
            channels = image_size[2]

            truebox_dict_list = []
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = str(obj.find('name').text)
                cls = cls.replace(' ', '').lower()
                if cls not in class_list:
                    continue
                if int(difficult) == 1:
                    continue
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                    float(xmlbox.find('ymax').text))
                xmin = max(min(float(b[0]), float(b[1]), float(width)), 0.)
                ymin = max(min(float(b[2]), float(b[3]), float(height)), 0.)
                xmax = min(max(float(b[1]), float(b[0]), 0.), float(width))
                ymax = min(max(float(b[3]), float(b[2]), 0.), float(height))
                truebox_dict_list.append(true_box(
                    cls, xmin, ymin, xmax, ymax, 'rectangle', difficult))  # 将单个真实框加入单张图片真实框列表
            one_image = per_image(image_name, img_path, int(
                height), int(width), int(channels), truebox_dict_list)
            total_images_data_list.append(one_image)    # 将单张图对象添加进全数据集数据列表中
        except:
            print("wrong: ", src_lab_dir)
            continue
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_hy_dataset(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'imageName'
    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    # 对应图片的source label文件路径
    src_lab_path = os.path.join(input_path, 'source_label')
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    total_images_data_list = []     # 声明全部图片列表
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        with open(src_lab_dir, 'r', encoding='unicode_escape') as f:
            data = js.load(f)
        for n, d in zip(range(len(data)), data):
            truebox_dict_list = []  # 声明每张图片真实框列表
            img_path = os.path.join(image_path, d[key])
            img = cv2.imread(img_path)
            height, width, channels = img.shape     # 读取每张图片的shape
            # 读取json文件中的每个真实框的class、xy信息
            if len(d["Data"]):
                for num, box in zip(range(len(d["Data"]["svgArr"])), d["Data"]["svgArr"]):
                    if box["tool"] == 'rectangle':
                        x = [float(box['data'][0]['x']), float(box['data'][1]['x']),
                             float(box['data'][2]['x']), float(box['data'][3]['x'])]
                        y = [float(box['data'][0]['y']), float(box['data'][1]['y']),
                             float(box['data'][2]['y']), float(box['data'][3]['y'])]
                        xmin = min(max(min(x), 0.), float(width))
                        ymin = min(max(min(y), 0.), float(height))
                        xmax = max(min(max(x), float(width)), 0.)
                        ymax = max(min(max(y), float(height)), 0.)
                        cls = box['secondaryLabel'][0]['value']
                        cls = cls.replace(' ', '').lower()
                        if cls not in class_list:
                            continue
                        truebox_dict_list.append(true_box(cls, min(x), min(y), max(
                            x), max(y), box["tool"]))  # 将单个真实框加入单张图片真实框列表
            one_image = per_image(d[key], img_path, int(
                height), int(width), int(channels), truebox_dict_list)
            total_images_data_list.append(one_image)    # 将单张图对象添加进全数据集数据列表中
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_hy_highway(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'filename'
    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(input_path, 'source_label')  # 对应图片的json文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    total_images_data_list = []     # 声明全部图片列表
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        tree = ET.parse(src_lab_dir)
        root = tree.getroot()
        image_name = str(root.find(key).text)
        img_path = img_path = os.path.join(image_path, image_name)
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        channels = int(size.find('depth').text)
        truebox_dict_list = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = str(obj.find('name').text)
            cls = cls.replace(' ', '').lower()
            if cls not in class_list:
                continue
            if int(difficult) == 1:
                continue
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            xmin = max(min(float(b[0]), float(b[1]), float(width)), 0.)
            ymin = max(min(float(b[2]), float(b[3]), float(height)), 0.)
            xmax = min(max(float(b[1]), float(b[0]), 0.), float(width))
            ymax = min(max(float(b[3]), float(b[2]), 0.), float(height))
            truebox_dict_list.append(true_box(
                cls, xmin, ymin, xmax, ymax, 'rectangle', difficult))  # 将单个真实框加入单张图片真实框列表
        one_image = per_image(image_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        total_images_data_list.append(one_image)    # 将单张图对象添加进全数据集数据列表中
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_pascal(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'filename'
    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(input_path, 'source_label')  # 对应图片的json文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    total_images_data_list = []     # 声明全部图片列表
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        tree = ET.parse(src_lab_dir)
        root = tree.getroot()
        image_name = str(root.find(key).text)
        img_path = img_path = os.path.join(image_path, image_name)
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        channels = int(size.find('depth').text)
        truebox_dict_list = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = str(obj.find('name').text)
            cls = cls.replace(' ', '').lower()
            if int(difficult) == 1:
                continue
            if cls not in class_list:
                continue
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            xmin = max(min(float(b[0]), float(b[1]), float(width)), 0.)
            ymin = max(min(float(b[2]), float(b[3]), float(height)), 0.)
            xmax = min(max(float(b[1]), float(b[0]), 0.), float(width))
            ymax = min(max(float(b[3]), float(b[2]), 0.), float(height))
            truebox_dict_list.append(true_box(
                cls, xmin, ymin, xmax, ymax, 'rectangle', difficult))  # 将单个真实框加入单张图片真实框列表
        one_image = per_image(image_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        total_images_data_list.append(one_image)    # 将单张图对象添加进全数据集数据列表中
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_kitti(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(input_path, 'source_label')  # 对应图片的json文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    total_images_data_list = []     # 声明全部图片列表
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        with open(src_lab_dir, 'r') as f:
            truebox_dict_list = []
            for one_bbox in f.readlines():
                one_bbox = one_bbox.strip('\n')
                bbox = one_bbox.split(' ')
                image_name = (src_lab_dir.split(
                    '\\')[-1]).replace('.txt', '.jpg')
                img_path = os.path.join(
                    image_path, image_name).replace('.txt', '.jpg')
                img = cv2.imread(img_path)
                size = img.shape
                width = int(size[1])
                height = int(size[0])
                channels = int(size[2])
                cls = str(one_bbox.split(' ')[0])
                cls = cls.strip(' ').lower()
                if cls == 'dontcare' or cls == 'misc':
                    continue
                if cls not in class_list:
                    continue
                xmin = min(
                    max(min(float(bbox[4]), float(bbox[6])), 0.), float(width))
                ymin = min(
                    max(min(float(bbox[5]), float(bbox[7])), 0.), float(height))
                xmax = max(
                    min(max(float(bbox[6]), float(bbox[4])), float(width)), 0.)
                ymax = max(
                    min(max(float(bbox[7]), float(bbox[5])), float(height)), 0.)
                truebox_dict_list.append(true_box(
                    cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
        one_image = per_image(image_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        total_images_data_list.append(one_image)    # 将单张图对象添加进全数据集数据列表中
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_coco_2017(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'file_name'
    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(input_path, 'source_label')  # 对应图片的json文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    total_images_data_list = []
    for src_lab_path_one in src_lab_path_list:
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        with open(src_lab_dir, 'r', encoding='unicode_escape') as f:
            data = js.loads(f)
        max_image_id = 0
        for one_image in data['images']:    # 获取数据集image中最大id数
            max_image_id = max(max_image_id, one_image['id'])
        for _ in range(max_image_id):   # 创建全图片列表
            total_images_data_list.append(None)
        name_dict = {}
        for one_name in data['categories']:
            name_dict['%s' % one_name['id']] = one_name['name']
        print('Start to load each annotation data file:')
        for d in tqdm(data['images']):   # 获取data字典中images内的图片信息，file_name、height、width
            img_id = d['id']
            img_name = d[key]
            img_path = os.path.join(image_path, d[key])
            img = cv2.imread(img_path)
            _, _, channels = img.shape
            width = d['width']
            height = d['height']
            # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
            # 并将初始化后的对象存入total_images_data_list
            one_image = per_image(
                img_name, img_path, height, width, channels, [])
            total_images_data_list[img_id - 1] = one_image
        for one_annotation in tqdm(data['annotations']):
            ann_image_id = one_annotation['image_id']   # 获取此bbox图片id
            cls = name_dict[str(one_annotation['category_id'])]     # 获取bbox类别
            cls = cls.replace(' ', '').lower()
            if cls not in class_list:
                continue
            # 将coco格式的bbox坐标转换为voc格式的bbox坐标，即xmin, xmax, ymin, ymax
            one_bbox_list = coco_voc(one_annotation['bbox'])
            # 为annotation对应的图片添加真实框信息
            one_bbox = true_box(cls,
                                min(max(float(one_bbox_list[0]), 0.), float(
                                    width)),
                                max(min(
                                    float(one_bbox_list[2]), float(height)), 0.),
                                min(max(float(one_bbox_list[1]), 0.), float(
                                    width)),
                                max(min(float(one_bbox_list[3]), float(height)), 0.))
            total_images_data_list[ann_image_id -
                                   1].true_box_list_updata(one_bbox)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_coco_2014(input_path, class_list):

    pass


def from_cctsdb(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(
        input_path, 'source_label')     # 对应图片的source label文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    total_images_data_list = []     # 声明全部图片列表
    image_name_dict = {}    # 声明图片名字典，用于判断图片是否已经添加至total_images_data_list
    image_index = 0     # 声明图片在total_images_data_list中的索引，用于添加bbox至total_images_data_list对应图片下
    image_erro = []     # 声明错误图片列表
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        with open(src_lab_dir, 'r') as f:
            for one_line in tqdm(f.readlines()):
                truebox_dict_list = []
                one_line = one_line.strip('\n')
                one_line_list = one_line.split(';')
                image_name = one_line_list[0]
                a = ''
                if a in one_line_list:
                    image_erro.append(image_name)
                    continue
                # TODO txt have repeat name
                if not(image_name in image_name_dict.keys()):
                    image_name_dict[image_name] = image_index
                    image_index += 1
                    img_path = os.path.join(image_path, image_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        image_erro.append(image_name)
                        del image_name_dict[image_name]
                        image_index -= 1
                        continue
                    size = img.shape
                    width = int(size[1])
                    height = int(size[0])
                    channels = int(size[2])
                    cls = str(one_line_list[5])
                    cls = cls.strip(' ').lower()
                    if cls not in class_list:
                        continue
                    xmin = min(max(min(float(one_line_list[1]), float(
                        one_line_list[3])), 0.), float(width))
                    ymin = min(max(min(float(one_line_list[4]), float(
                        one_line_list[2])), 0.), float(height))
                    xmax = max(min(max(float(one_line_list[3]), float(
                        one_line_list[1])), float(width)), 0.)
                    ymax = max(min(max(float(one_line_list[2]), float(
                        one_line_list[4])), float(height)), 0.)
                    truebox_dict_list.append(true_box(
                        cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
                    one_image = per_image(image_name, img_path, int(
                        height), int(width), int(channels), truebox_dict_list)
                    # 将单张图对象添加进全数据集数据列表中
                    total_images_data_list.append(one_image)
                else:
                    img_path = os.path.join(image_path, image_name)
                    img = cv2.imread(img_path)
                    size = img.shape
                    width = int(size[1])
                    height = int(size[0])
                    channels = int(size[2])
                    cls = str(one_line_list[5])
                    cls = cls.strip(' ').lower()
                    if cls not in class_list:
                        continue
                    xmin = min(max(min(float(one_line_list[1]), float(
                        one_line_list[3])), 0.), float(width))
                    ymin = min(max(min(float(one_line_list[4]), float(
                        one_line_list[2])), 0.), float(height))
                    xmax = max(min(max(float(one_line_list[3]), float(
                        one_line_list[1])), float(width)), 0.)
                    ymax = max(min(max(float(one_line_list[2]), float(
                        one_line_list[4])), float(height)), 0.)
                    total_images_data_list[-1].true_box_list.append(true_box(
                        cls, xmin, ymin, xmax, ymax))
    print('Total: %d images extract, Done!' % len(total_images_data_list))
    for a in image_erro:
        print('\n%s erro!' % a)

    return total_images_data_list


def from_lisa(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(
        input_path, 'source_label')     # 对应图片的source label文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    total_images_data_list = []     # 声明全部图片列表
    image_name_dict = {}    # 声明图片名字典，用于判断图片是否已经添加至total_images_data_list
    image_index = 0     # 声明图片在total_images_data_list中的索引，用于添加bbox至total_images_data_list对应图片下
    image_erro = []     # 声明错误图片列表
    # stop_list = ['stop', 'stopleft', 'warning', 'warningleft']
    # go_list = ['go', 'goleft']
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        with open(src_lab_dir, 'r') as f:
            read_csv = csv.reader(f)
            for one_line in tqdm(read_csv):
                truebox_dict_list = []
                image_name = one_line[0]
                # 判断图片是否已经记录真实框
                if not(image_name in image_name_dict.keys()):
                    image_name_dict[image_name] = image_index
                    image_index += 1
                    img_path = os.path.join(image_path, image_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        image_erro.append(image_name)
                        del image_name_dict[image_name]
                        image_index -= 1
                        continue
                    # 获取图片信息
                    size = img.shape
                    width = int(size[1])
                    height = int(size[0])
                    channels = int(size[2])
                    # 获取真实框信息
                    cls = str(one_line[1])
                    cls = cls.strip(' ').lower()
                    if cls not in class_list:
                        continue
                    xmin = min(
                        max(min(float(one_line[2]), float(one_line[4])), 0.), float(width))
                    ymin = min(
                        max(min(float(one_line[3]), float(one_line[5])), 0.), float(height))
                    xmax = max(
                        min(max(float(one_line[2]), float(one_line[4])), float(width)), 0.)
                    ymax = max(
                        min(max(float(one_line[3]), float(one_line[5])), float(height)), 0.)
                    truebox_dict_list.append(true_box(
                        cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
                    one_image = per_image(image_name, img_path, int(
                        height), int(width), int(channels), truebox_dict_list)
                    # 将单张图对象添加进全数据集数据列表中
                    total_images_data_list.append(one_image)
                else:
                    img_path = os.path.join(image_path, image_name)
                    img = cv2.imread(img_path)
                    # 获取图片信息
                    size = img.shape
                    width = int(size[1])
                    height = int(size[0])
                    channels = int(size[2])
                    # 获取真实框信息
                    cls = str(one_line[1])
                    cls = cls.strip(' ').lower()
                    if cls not in class_list:
                        continue
                    xmin = min(
                        max(min(float(one_line[2]), float(one_line[4])), 0.), float(width))
                    ymin = min(
                        max(min(float(one_line[3]), float(one_line[5])), 0.), float(height))
                    xmax = max(
                        min(max(float(one_line[2]), float(one_line[4])), float(width)), 0.)
                    ymax = max(
                        min(max(float(one_line[3]), float(one_line[5])), float(height)), 0.)
                    one_image = per_image(image_name, img_path, int(
                        height), int(width), int(channels), truebox_dict_list)
                    # 将单张图对象添加进全数据集数据列表中
                    total_images_data_list.append(one_image)

    print('Total: %d images extract, Done!' % len(total_images_data_list))
    if len(image_erro) != 0:
        for a in image_erro:
            print('\n%s erro!' % a)

    return total_images_data_list


def from_yolo(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(
        input_path, 'source_label')     # 对应图片的source label文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    names_list_path = get_names_list_path(input_path)
    class_list = get_class(names_list_path)
    total_images_data_list = []     # 声明全部图片列表
    image_name_dict = {}    # 声明图片名字典，用于判断图片是否已经添加至total_images_data_list
    image_index = 0     # 声明图片在total_images_data_list中的索引，用于添加bbox至total_images_data_list对应图片下
    image_erro = []     # 声明错误图片列表
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        with open(src_lab_dir, 'r') as f:
            truebox_dict_list = []
            for one_bbox in f.readlines():
                one_bbox = one_bbox.strip('\n')
                bbox = one_bbox.split(' ')[1:]
                image_name = (src_lab_dir.split(
                    '/')[-1]).replace('.txt', '.jpg')
                img_path = os.path.join(
                    image_path, image_name)
                img = cv2.imread(img_path)
                if img is None:
                    image_erro.append(image_name)
                    del image_name_dict[image_name]
                    image_index -= 1
                    continue
                size = img.shape
                width = int(size[1])
                height = int(size[0])
                channels = int(size[2])
                cls = class_list[int(one_bbox.split(' ')[0])]
                cls = cls.strip(' ').lower()
                if cls not in class_list:
                    continue
                if cls == 'dontcare' or cls == 'misc':
                    continue
                bbox = revers_yolo(size, bbox)
                xmin = min(
                    max(min(float(bbox[0]), float(bbox[1])), 0.), float(width))
                ymin = min(
                    max(min(float(bbox[2]), float(bbox[3])), 0.), float(height))
                xmax = max(
                    min(max(float(bbox[1]), float(bbox[0])), float(width)), 0.)
                ymax = max(
                    min(max(float(bbox[3]), float(bbox[2])), float(height)), 0.)
                truebox_dict_list.append(true_box(
                    cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
        one_image = per_image(image_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        # 将单张图对象添加进全数据集数据列表中
        total_images_data_list.append(one_image)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_sjt(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    def change_traffic_light(truebox_dict_list):
        """[修改数据堂信号灯标签信息，将灯与信号灯框结合]

        Parameters
        ----------
        truebox_dict_list : [list]
            [源真实框]

        Returns
        -------
        new_truebox_dict_list : [list]
            [修改后真实框]
        """
        light_name = ['ordinarylight',
                      'goingstraight',
                      'turningleft',
                      'turningright',
                      'u-turn',
                      'u-turn&turningleft',
                      'turningleft&goingstraight',
                      'turningright&goingstraight',
                      'u-turn&goingstraight',
                      'numbers']
        light_base_name = ['trafficlightframe_'+i for i in light_name]
        light_base_name.append('trafficlightframe')
        light_go = []   # 声明绿灯信号灯命名列表
        light_stop = []     # 声明红灯信号灯命名列表
        light_warning = []      # 声明黄灯信号灯命名列表
        for one_name in light_base_name:
            light_go.append(one_name + '_' + 'green')
        for one_name in light_base_name:
            light_stop.append(one_name + '_' + 'red')
        for one_name in light_base_name:
            light_stop.append(one_name + '_' + 'yellow')
            light_stop.append(one_name + '_' + 'unclear')
            light_stop.append(one_name + '_' + 'no')
        light_numbers = []
        light_numbers.append('trafficlightframe_numbers' + '_' + 'green')
        light_numbers.append('trafficlightframe_numbers' + '_' + 'red')
        light_numbers.append('trafficlightframe_numbers' + '_' + 'yellow')
        light_numbers.append('trafficlightframe_numbers' + '_' + 'unclear')
        light_numbers.append('trafficlightframe_numbers' + '_' + 'no')
        new_truebox_dict_list = []  # 声明新真实框列表
        for one_true_box in truebox_dict_list:  # 遍历源真实框列表
            if one_true_box.clss == 'trafficlightframe':    # 搜索trafficlightframe真实框
                if one_true_box.color == 'no':
                    for light_true_box in truebox_dict_list:    # 遍历源真实框列表
                        if (light_true_box.clss in light_name and
                            light_true_box.xmin >= one_true_box.xmin - 20 and
                            light_true_box.ymin >= one_true_box.ymin - 20 and
                            light_true_box.xmax <= one_true_box.xmax + 20 and
                                light_true_box.ymax <= one_true_box.ymax + 20):  # 判断信号灯框类别
                            one_true_box.clss += (
                                '_' + light_true_box.clss + '_' + light_true_box.color)   # 新建信号灯真实框实例并更名
                        if one_true_box.clss in light_numbers:
                            continue
                        if one_true_box.clss in light_go:     # 将信号灯归类
                            one_true_box.clss = 'go'
                        if one_true_box.clss in light_stop:
                            one_true_box.clss = 'stop'
                        if one_true_box.clss in light_warning:
                            one_true_box.clss = 'warning'
                    if one_true_box.clss == 'trafficlightframe':    # 若为发现框内有信号灯颜色则更换为warning
                        one_true_box.clss = 'warning'
                else:
                    innate_light = ''
                    for light_true_box in truebox_dict_list:    # 遍历源真实框列表
                        if (light_true_box.clss in light_name and
                            light_true_box.xmin >= one_true_box.xmin - 20 and
                            light_true_box.ymin >= one_true_box.ymin - 20 and
                            light_true_box.xmax <= one_true_box.xmax + 20 and
                                light_true_box.ymax <= one_true_box.ymax + 20):  # 判断信号灯框类别
                            innate_light = light_true_box.clss
                    if innate_light == 'numbers':
                        continue
                    one_true_box.clss += ('_' + one_true_box.color)
                    if one_true_box.clss in light_go:     # 将信号灯归类
                        one_true_box.clss = 'go'
                    if one_true_box.clss in light_stop:
                        one_true_box.clss = 'stop'
                    if one_true_box.clss in light_warning:
                        one_true_box.clss = 'warning'
            # if one_true_box.clss == 'numbers':
            #     if one_true_box.color == 'green':     # 将数字类信号灯归类
            #         one_true_box.clss = 'go'
            #     if one_true_box.color == 'red':
            #         one_true_box.clss = 'stop'
            #     if (one_true_box.color == 'yellow' or
            #         one_true_box.color == 'no' or
            #         one_true_box.color == 'unclear'):
            #         one_true_box.clss = 'warning'
            if one_true_box.clss in class_list:
                new_truebox_dict_list.append(one_true_box)

        return new_truebox_dict_list

    def change_Occlusion(source_occlusion):
        """[转换真实框遮挡信息]

        Args:
            source_occlusion ([str]): [ture box遮挡信息]

        Returns:
            [int]: [返回遮挡值]
        """

        occlusion = 0
        if source_occlusion == "No occlusion (0%)":
            occlusion = 0
        if source_occlusion == "Partial occlusion (0%~35%)":
            occlusion = 35
        if source_occlusion == "Occlusion for most parts (35%~50%)":
            occlusion = 50
        if source_occlusion == "Others (more than 50%)":
            occlusion = 75

        return occlusion

    key = 'imageName'
    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    # 对应图片的source label文件路径
    src_lab_path = os.path.join(input_path, 'source_label')
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    total_images_data_list = []     # 声明全部图片列表
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        with open(src_lab_dir, 'r', encoding='unicode_escape') as f:
            data = js.load(f)
        img_name = src_lab_path_one.split('.')[0] + '.jpg'
        img_path = os.path.join(image_path, img_name)
        img = cv2.imread(img_path)
        height, width, channels = img.shape     # 读取每张图片的shape
        truebox_dict_list = []  # 声明每张图片真实框列表
        if len(data["boxs"]):
            for one_box in data["boxs"]:
                # 读取json文件中的每个真实框的class、xy信息
                x = one_box['x']
                y = one_box['y']
                xmin = min(max(float(x), 0.), float(width))
                ymin = min(max(float(y), 0.), float(height))
                xmax = max(min(float(x + one_box['w']), float(width)), 0.)
                ymax = max(min(float(y + one_box['h']), float(height)), 0.)
                true_box_color = ''
                cls = ''
                types = one_box['Object_type']
                types = types.replace(' ', '').lower()
                if types == 'pedestrians' or types == 'vehicles' or types == 'trafficlights':
                    cls = one_box['Category']
                    if types == 'trafficlights':    # 获取交通灯颜色
                        true_box_color = one_box['Color']
                        true_box_color = true_box_color.replace(
                            ' ', '').lower()
                else:
                    cls = types
                cls = cls.replace(' ', '').lower()
                if cls == 'misc' or cls == 'dontcare':
                    continue
                ture_box_occlusion = 0
                if 'Occlusion' in one_box:
                    ture_box_occlusion = change_Occlusion(
                        one_box['Occlusion'])  # 获取真实框遮挡率
                ture_box_distance = 0
                if 'distance' in one_box:
                    ture_box_distance = one_box['distance']  # 获取真实框中心点距离
                if xmax > xmin and ymax > ymin:
                    # 将单个真实框加入单张图片真实框列表
                    truebox_dict_list.append(
                        true_box(cls, xmin, ymin, xmax, ymax, true_box_color, 0, occlusion=ture_box_occlusion, distance=ture_box_distance))
                else:
                    print('\nBbox error!')
                    continue
        truebox_dict_list = change_traffic_light(
            truebox_dict_list)  # 添加修改信号灯框名称后的真实框
        for one_true_box in truebox_dict_list:
            if one_true_box.clss not in class_list:
                truebox_dict_list.pop(truebox_dict_list.index(one_true_box))
        one_image = per_image(img_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        # 将单张图对象添加进全数据集数据列表中
        total_images_data_list.append(one_image)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list

# TODO


def from_nuscenes(input_path, class_list):

    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(
        input_path, 'source_label')     # 对应图片的source label文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    names_list_path = get_names_list_path(input_path)
    class_list = get_class(names_list_path)
    total_images_data_list = []     # 声明全部图片列表
    image_name_dict = {}    # 声明图片名字典，用于判断图片是否已经添加至total_images_data_list
    image_index = 0     # 声明图片在total_images_data_list中的索引，用于添加bbox至total_images_data_list对应图片下
    image_erro = []     # 声明错误图片列表
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)

        print(0)
        pass

    return total_images_data_list


def from_yolov5(input_path, class_list):

    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(
        input_path, 'source_label')     # 对应图片的source label文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    names_list_path = get_names_list_path(input_path)
    class_list = get_class(names_list_path)

    total_images_data_list = []     # 声明全部图片列表
    image_name_dict = {}    # 声明图片名字典，用于判断图片是否已经添加至total_images_data_list
    image_index = 0     # 声明图片在total_images_data_list中的索引，用于添加bbox至total_images_data_list对应图片下
    image_erro = []     # 声明错误图片列表
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        with open(src_lab_dir, 'r') as f:
            truebox_dict_list = []
            for one_bbox in f.readlines():
                one_bbox = one_bbox.strip('\n')
                bbox = one_bbox.split(' ')[1:]
                image_name = (src_lab_dir.split(
                    '/')[-1]).replace('.txt', '.jpg')
                img_path = os.path.join(
                    image_path, image_name).replace('.txt', '.jpg')
                img = cv2.imread(img_path)
                if img is None:
                    image_erro.append(image_name)
                    del image_name_dict[image_name]
                    image_index -= 1
                    continue
                size = img.shape
                width = int(size[1])
                height = int(size[0])
                channels = int(size[2])
                cls = class_list[int(one_bbox.split(' ')[0])]
                cls = cls.strip(' ').lower()
                if cls not in class_list:
                    continue
                if cls == 'dontcare' or cls == 'misc':
                    continue
                bbox = revers_yolo(size, bbox)
                xmin = min(
                    max(min(float(bbox[0]), float(bbox[1])), 0.), float(width))
                ymin = min(
                    max(min(float(bbox[2]), float(bbox[3])), 0.), float(height))
                xmax = max(
                    min(max(float(bbox[1]), float(bbox[0])), float(width)), 0.)
                ymax = max(
                    min(max(float(bbox[3]), float(bbox[2])), float(height)), 0.)
                truebox_dict_list.append(true_box(
                    cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
        one_image = per_image(image_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        # 将单张图对象添加进全数据集数据列表中
        total_images_data_list.append(one_image)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_ccpd(input_path, class_list):

    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(
        input_path, 'source_label')     # 对应图片的source label文件路径
    total_label_list = []
    total_images_data_list = []     # 声明全部图片列表
    image_erro = []     # 声明错误图片列表

    for one_label in tqdm(os.listdir(src_lab_path)):
        with open(os.path.join(src_lab_path, one_label), 'r') as f:
            for n in f.readlines():
                truebox_dict_list = []
                one_bbox = n.strip('\n')
                image_name = one_bbox
                bbox_info = one_bbox.strip('ccpd_').strip('.jpg').split('-')
                img_path = os.path.join(
                    image_path, image_name)
                img = cv2.imread(img_path)
                if img is None:
                    image_erro.append(image_name)
                    continue
                size = img.shape
                width = int(size[1])
                height = int(size[0])
                channels = int(size[2])
                cls = 'licenseplate'
                if cls not in class_list:
                    continue
                bbox = [int(bbox_info[2].split('_')[0].split('#')[0]),
                        int(bbox_info[2].split('_')[0].split('#')[1]),
                        int(bbox_info[2].split('_')[1].split('#')[0]),
                        int(bbox_info[2].split('_')[1].split('#')[1])]
                xmin = min(
                    max(min(float(bbox[0]), float(bbox[2])), 0.), float(width))
                ymin = min(
                    max(min(float(bbox[1]), float(bbox[3])), 0.), float(height))
                xmax = max(
                    min(max(float(bbox[0]), float(bbox[2])), float(width)), 0.)
                ymax = max(
                    min(max(float(bbox[1]), float(bbox[3])), float(height)), 0.)
                truebox_dict_list.append(true_box(
                    cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
                one_image = per_image(image_name, img_path, int(
                    height), int(width), int(channels), truebox_dict_list)
                # 将单张图对象添加进全数据集数据列表中
                total_images_data_list.append(one_image)

    return total_images_data_list


def from_licenseplate(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""
    
    local_mask = {"皖": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9,
                  "苏": 10, "浙": 11, "京": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18,
                  "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24, "西": 25, "陕": 26, "甘": 27,
                  "青": 28, "宁": 29, "新": 30}
    
    code_mask = {"a" : 0, "b" : 1, "c" : 2, "d" : 3, "e" : 4, "f" : 5, "g" : 6, "h" : 7, "j" : 8, "k" : 9,
                 "l" : 10, "m" : 11, "n" : 12, "p" : 13, "q" : 14, "r" : 15, "s" : 16, "t" : 17, "u" : 18,
                 "v" : 19, "w" : 20, "x":  21, "y" : 22, "z" : 23, "0_" : 24, "1" : 25, "2" : 26, "3" : 27,
                 "4" : 28, "5" : 29, "6" : 30, "7" : 31, "8" : 32, "9" : 33}
    
    local_mask_key_list = [x for x, _ in local_mask.items()]
    code_mask_key_list = [x for x, _ in code_mask.items()]
    
    image_path = os.path.join(input_path, 'JPEGImages')   # 图片路径
    src_lab_path = os.path.join(
        input_path, 'source_label')     # 对应图片的source label文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
    names_list_path = get_names_list_path(input_path)
    class_list = get_class(names_list_path)
    total_images_data_list = []     # 声明全部图片列表
    image_name_dict = {}    # 声明图片名字典，用于判断图片是否已经添加至total_images_data_list
    image_index = 0     # 声明图片在total_images_data_list中的索引，用于添加bbox至total_images_data_list对应图片下
    image_erro = []     # 声明错误图片列表
    print('Start to load each annotation data file:')
    # 将每一个source label文件转换为per_image类
    for src_lab_path_one in tqdm(src_lab_path_list):
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        with open(src_lab_dir, 'r') as f:
            truebox_dict_list = []
            for one_bbox in f.readlines():
                one_bbox = one_bbox.strip('\n')
                bbox = one_bbox.split(' ')[1:]
                image_name = (src_lab_dir.split(
                    '/')[-1]).replace('.txt', '.jpg')
                img_path = os.path.join(
                    image_path, image_name)
                img = cv2.imread(img_path)
                if img is None:
                    image_erro.append(image_name)
                    del image_name_dict[image_name]
                    image_index -= 1
                    continue
                size = img.shape
                width = int(size[1])
                height = int(size[0])
                channels = int(size[2])
                cls = class_list[int(one_bbox.split(' ')[0])]
                cls = cls.strip(' ').lower()
                if cls not in class_list:
                    continue
                if cls == 'dontcare' or cls == 'misc':
                    continue
                bbox = revers_yolo(size, bbox)
                xmin = min(
                    max(min(float(bbox[0]), float(bbox[1])), 0.), float(width))
                ymin = min(
                    max(min(float(bbox[2]), float(bbox[3])), 0.), float(height))
                xmax = max(
                    min(max(float(bbox[1]), float(bbox[0])), float(width)), 0.)
                ymax = max(
                    min(max(float(bbox[3]), float(bbox[2])), float(height)), 0.)
                truebox_dict_list.append(true_box(
                    cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
        if 7 != len(truebox_dict_list):
            continue
        truebox_dict_list.sort(key=lambda x:x.xmin)
        
        # 更换真实框类别为车牌真实值
        real_classes_list = list(map(int,src_lab_path_one.split('-')[4].split('_')))
        classes_decode_list = []
        classes_decode_list.append(local_mask_key_list[real_classes_list[0]])
        for one in real_classes_list[1:]:
            classes_decode_list.append(code_mask_key_list[one])
        for truebox, classes in zip(truebox_dict_list, classes_decode_list):
            truebox.clss = classes
        one_image = per_image(image_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        # 将单张图对象添加进全数据集数据列表中
        total_images_data_list.append(one_image)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def to_pascal(output_path, images_data_list):
    """将输入标签转换后的图片字典列表，存储为voc的xml格式文件。"""

    print('\nStart to write each label data file:')
    xml_count = 0
    for image_data in tqdm(images_data_list):
        if image_data == None:
            continue
        with codecs.open(os.path.join(output_path, image_data.image_name.split('.')[0] + ".xml"), "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
            xml.write('\t<filename>' + image_data.image_name + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>The VOC2007 Database</database>\n')
            xml.write('\t\t<annotation>PASCAL VOC2007</annotation>\n')
            xml.write('\t\t<image>flickr</image>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t</source>\n')
            xml.write('\t<owner>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t\t<name>WH</name>\n')
            xml.write('\t</owner>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(int(image_data.width)) + '</width>\n')
            xml.write('\t\t<height>' +
                      str(int(image_data.height)) + '</height>\n')
            xml.write('\t\t<depth>' +
                      str(int(image_data.channels)) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')
            if len(image_data.true_box_list):
                for box in image_data.true_box_list:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + box.clss + '</name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    xml.write('\t\t<truncated>1</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' +
                              str(int(box.xmin)) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' +
                              str(int(box.ymin)) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' +
                              str(int(box.xmax)) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' +
                              str(int(box.ymax)) + '</ymax>\n')
                    # TODO: 距离和遮挡                              
                    # xml.write('\t\t\t<distance>' +
                    #           str(int(box.distance)) + '</distance>\n')
                    # xml.write('\t\t\t<occlusion>' +
                    #           str(float(box.occlusion)) + '</occlusion>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')
            xml.write('</annotation>')
        xml_count += 1
        # print(image_data.image_path + " has saved in " + xml_path)
    print('Total: %d file change save to %s, Done!' % (xml_count, output_path))
    
    
def to_coco(output_path, images_data_list):
    print('\nStart to write each label data file:')
    xml_count = 0
    for image_data in tqdm(images_data_list):
        if image_data == None:
            continue
        with codecs.open(os.path.join(output_path, image_data.image_name.split('.')[0] + ".xml"), "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
            xml.write('\t<filename>' + image_data.image_name + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>The VOC2007 Database</database>\n')
            xml.write('\t\t<annotation>PASCAL VOC2007</annotation>\n')
            xml.write('\t\t<image>flickr</image>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t</source>\n')
            xml.write('\t<owner>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t\t<name>WH</name>\n')
            xml.write('\t</owner>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(int(image_data.width)) + '</width>\n')
            xml.write('\t\t<height>' +
                      str(int(image_data.height)) + '</height>\n')
            xml.write('\t\t<depth>' +
                      str(int(image_data.channels)) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')
            if len(image_data.true_box_list):
                for box in image_data.true_box_list:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + box.clss + '</name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    xml.write('\t\t<truncated>1</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' +
                              str(int(box.xmin)) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' +
                              str(int(box.ymin)) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' +
                              str(int(box.xmax)) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' +
                              str(int(box.ymax)) + '</ymax>\n')
                    # TODO: 距离和遮挡
                    # xml.write('\t\t\t<distance>' +
                    #           str(int(box.distance)) + '</distance>\n')
                    # xml.write('\t\t\t<occlusion>' +
                    #           str(float(box.occlusion)) + '</occlusion>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')
            xml.write('</annotation>')
        xml_count += 1
        # print(image_data.image_path + " has saved in " + xml_path)
    print('Total: %d file change save to %s, Done!' % (xml_count, output_path))


def random_test(pre_data_list):
    """[为无距离、遮挡属性的数据添加随机距离和遮挡率]

    Args:
        pre_data_list ([list]): [读取的图片类信息列表]

    Returns:
        [list]: [随机修改距离和遮挡率的图片类信息列表]
    """

    distance_list = [0, 50, 100]
    occlusion_list = [0, 35, 50, 75]
    data_list = pre_data_list.copy()
    for one_image_info in data_list:
        for one_true_box in one_image_info.true_box_list:
            one_true_box.distance = random.sample(distance_list, 1)[0]
            one_true_box.occlusion = random.sample(occlusion_list, 1)[0]
    return data_list


def in_func_None(*args):
    """如无对应输入标签函数，提示用户添加输入标签函数"""
    print("\n如无对应输入标签函数，请添加输入标签函数。")
    return 0


in_func_dict = {"ldp": from_ldp, "hy": from_hy_dataset, "myxb": from_hy_dataset, "hy_highway": from_hy_highway,
                "pascal": from_pascal, "kitti": from_kitti, "coco2017": from_coco_2017,
                "cctsdb": from_cctsdb, "lisa": from_lisa, "yolo": from_yolo, "hanhe": from_yolo,
                "sjt": from_sjt, "nuscenes": from_nuscenes, "yolov5_detect": from_yolov5, "ccpd": from_ccpd,
                "licenseplate": from_licenseplate}
out_func_dict = {"ldp": to_pascal, "pascal": to_pascal,
                 "coco": to_coco
                 }


def pickup_data_from_function(input_label_style, *args):
    """根据输入类别挑选数据集提取、转换函数"""
    # 返回对应数据获取函数
    return in_func_dict.get(input_label_style, in_func_None)(*args)


def pickup_data_out_function(output_label_style, *args):
    """根据输入类别挑选转换函数"""
    # 返回对应数据输出函数
    return out_func_dict.get(output_label_style, out_func_None)(*args)


def delete_other_class(total_data_list, one_class):
    """[删除其他非指定类别图片信息]

    Args:
        total_data_list ([list]): [总图片信息列表]
        one_class ([str]): [指定保留类别]

    Returns:
        [list]: [删除其他非指定类别图片信息后的总图片列表]
    """

    one_class_set = {one_class}
    new_total_data_list = []
    print('\nStart to delete images which not only\t%s:' % one_class)
    for one_image in tqdm(total_data_list):
        ture_boxes_set = []
        for one_bbox in one_image.true_box_list:
            ture_boxes_set.append(one_bbox.clss)
        ture_boxes_set = set(ture_boxes_set)
        if not ture_boxes_set.issubset(one_class_set):
            continue
        else:
            new_total_data_list.append(one_image)

    return new_total_data_list
