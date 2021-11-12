# -*- coding: utf-8 -*-
import os
import codecs
import json as js
import cv2
from tqdm import tqdm
import csv
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from utils.utils import *
from utils.convertion_function import *


class true_box:
    """真实框类"""

    def __init__(self, clss, xmin, ymin, xmax, ymax, tool='', difficult=0):
        self.clss = clss
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.tool = tool    # bbox工具
        self.difficult = difficult


class per_image:
    """图片类"""

    def __init__(self, image_name_in, image_path_in, height_in, width_in, channels_in, true_box_list_in):
        self.image_name = image_name_in    # 图片名称
        self.image_path = image_path_in    # 图片地址
        self.height = height_in    # 图片高
        self.width = width_in  # 图片宽
        self.channels = channels_in    # 图片通道数
        self.true_box_list = true_box_list_in  # 图片真实框列表
        self.mask = 0    # 获取图片的真实框掩码图

    def get_true_box_mask(self):
        """[获取图片真实框掩码图，前景置1，背景置0]

        Parameters
        ----------
        true_box_list : [list]
            [真实框列表]
        """
        
        self.mask = np.zeros((self.height, self.width))
        for one_ture_box in self.true_box_list:     # 读取true_box并对前景在mask上置1
            self.mask[int(one_ture_box.ymin):int(one_ture_box.ymax),
                      int(one_ture_box.xmin):int(one_ture_box.xmax)] = 1     # 将真实框范围内置1

    def true_box_list_updata(self, one_bbox_data):
        """[为per_image对象true_box_list成员添加元素]

        Parameters
        ----------
        one_bbox_data : [class true_box]
            [真实框类]
        """

        self.true_box_list.append(one_bbox_data)

    # TODO
    def get_free_space_circle_center_point(self, mask):

        pass

    # TODO
    def get_free_space_circle(self, free_space_circle_center_point):

        pass


def from_ldp(input_path, class_list):
    """ldp格式，抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'filename'
    image_path = os.path.join(input_path, 'images')   # 图片路径
    src_lab_path = os.path.join(input_path, 'Annotations')  # 对应图片的json文件路径
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
            cls = cls.strip(' ').lower()
            if cls not in class_list:
                continue
            if int(difficult) == 1:
                continue
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            xmin = float(min(float(b[0]), float(b[1])))
            ymin = float(min(float(b[2]), float(b[3])))
            xmax = float(max(float(b[1]), float(b[0])))
            ymax = float(max(float(b[3]), float(b[2])))
            truebox_dict_list.append(true_box(
                cls, xmin, ymin, xmax, ymax, 'rectangle', difficult))  # 使用动态名称变量将单个真实框加入单张图片真实框列表
        one_image = per_image(image_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
        # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
        total_images_data_list.append(one_image)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_hy_dataset(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'imageName'
    image_path = os.path.join(input_path, 'images')   # 图片路径
    # 对应图片的source label文件路径
    src_lab_path = os.path.join(input_path, 'source label')
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
                        xmin = min(x)
                        ymin = min(y)
                        xmax = max(x)
                        ymax = max(y)
                        cls = box['secondaryLabel'][0]['value']
                        cls = cls.replace(' ', '').lower()
                        if cls not in class_list:
                            continue
                        true_box_name = locals()    # 定义true_box_name为局部变量
                        if xmax > xmin and ymax > ymin:
                            truebox_dict_list.append(true_box(cls, min(x), min(y), max(
                                x), max(y), box["tool"]))  # 使用动态名称变量将单个真实框加入单张图片真实框列表
            one_image = per_image(d[key], img_path, int(
                height), int(width), int(channels), truebox_dict_list)
            # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
            # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
            total_images_data_list.append(one_image)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_hy_highway(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'filename'
    image_path = os.path.join(input_path, 'images')   # 图片路径
    src_lab_path = os.path.join(input_path, 'source label')  # 对应图片的json文件路径
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
            xmin = float(min(float(b[0]), float(b[1])))
            ymin = float(min(float(b[2]), float(b[3])))
            xmax = float(max(float(b[1]), float(b[0])))
            ymax = float(max(float(b[3]), float(b[2])))
            truebox_dict_list.append(true_box(
                cls, xmin, ymin, xmax, ymax, 'rectangle', difficult))  # 使用动态名称变量将单个真实框加入单张图片真实框列表


        one_image = per_image(image_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
        # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
        total_images_data_list.append(one_image)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_pascal(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'filename'
    image_path = os.path.join(input_path, 'images')   # 图片路径
    src_lab_path = os.path.join(input_path, 'source label')  # 对应图片的json文件路径
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
            xmin = float(min(float(b[0]), float(b[1])))
            ymin = float(min(float(b[2]), float(b[3])))
            xmax = float(max(float(b[1]), float(b[0])))
            ymax = float(max(float(b[3]), float(b[2])))
            truebox_dict_list.append(true_box(
                cls, xmin, ymin, xmax, ymax, 'rectangle', difficult))  # 使用动态名称变量将单个真实框加入单张图片真实框列表
        one_image = per_image(image_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
        # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
        total_images_data_list.append(one_image)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_kitti(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    image_path = os.path.join(input_path, 'images')   # 图片路径
    src_lab_path = os.path.join(input_path, 'source label')  # 对应图片的json文件路径
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
                    '\\')[-1]).replace('.txt', '.png')
                img_path = os.path.join(
                    image_path, image_name).replace('.txt', '.png')
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
                xmin = float(min(float(bbox[4]), float(bbox[6])))
                ymin = float(min(float(bbox[5]), float(bbox[7])))
                xmax = float(max(float(bbox[6]), float(bbox[4])))
                ymax = float(max(float(bbox[7]), float(bbox[5])))
                truebox_dict_list.append(true_box(
                    cls, xmin, ymin, xmax, ymax))  # 使用动态名称变量将单个真实框加入单张图片真实框列表
        one_image = per_image(image_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
        # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
        total_images_data_list.append(one_image)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list


def from_coco(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'file_name'

    image_path = os.path.join(input_path, 'images')   # 图片路径
    src_lab_path = os.path.join(input_path, 'source label')  # 对应图片的json文件路径
    src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名

    total_images_data_list = []
    for src_lab_path_one in src_lab_path_list:
        src_lab_dir = os.path.join(src_lab_path, src_lab_path_one)
        with open(src_lab_dir, 'r', encoding='unicode_escape') as f:
            data = js.load(f)
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
            # one_image.get_true_box_mask()   # 获取每张图片的真实框掩码图
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
                                one_bbox_list[0],
                                one_bbox_list[2],
                                one_bbox_list[1],
                                one_bbox_list[3])
            total_images_data_list[ann_image_id -
                                   1].true_box_list_updata(one_bbox)
        # for one in total_images_data_list:
        #     one.get_true_box_mask()     # 获取每张图片的真实框掩码图
        
    return total_images_data_list


def from_cctsdb(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    image_path = os.path.join(input_path, 'images')   # 图片路径
    src_lab_path = os.path.join(
        input_path, 'source label')     # 对应图片的source label文件路径
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
                    xmin = float(
                        min(float(one_line_list[1]), float(one_line_list[3])))
                    ymin = float(
                        min(float(one_line_list[4]), float(one_line_list[2])))
                    xmax = float(
                        max(float(one_line_list[3]), float(one_line_list[1])))
                    ymax = float(
                        max(float(one_line_list[2]), float(one_line_list[4])))
                    truebox_dict_list.append(true_box(
                        cls, xmin, ymin, xmax, ymax))  # 使用动态名称变量将单个真实框加入单张图片真实框列表
                    one_image = per_image(image_name, img_path, int(
                        height), int(width), int(channels), truebox_dict_list)
                    # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
                    # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
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
                    xmin = float(
                        min(float(one_line_list[1]), float(one_line_list[3])))
                    ymin = float(
                        min(float(one_line_list[4]), float(one_line_list[2])))
                    xmax = float(
                        max(float(one_line_list[3]), float(one_line_list[1])))
                    ymax = float(
                        max(float(one_line_list[2]), float(one_line_list[4])))
                    one_image = per_image(image_name, img_path, int(
                        height), int(width), int(channels), truebox_dict_list)
                    # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
                    # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
                    total_images_data_list.append(one_image)

    print('Total: %d images extract, Done!' % len(total_images_data_list))
    for a in image_erro:
        print('\n%s erro!' % a)

    return total_images_data_list


def from_lisa(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    image_path = os.path.join(input_path, 'images')   # 图片路径
    src_lab_path = os.path.join(
        input_path, 'source label')     # 对应图片的source label文件路径
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
                    xmin = float(min(float(one_line[2]), float(one_line[4])))
                    ymin = float(min(float(one_line[3]), float(one_line[5])))
                    xmax = float(max(float(one_line[2]), float(one_line[4])))
                    ymax = float(max(float(one_line[3]), float(one_line[5])))
                    truebox_dict_list.append(true_box(
                        cls, xmin, ymin, xmax, ymax))  # 使用动态名称变量将单个真实框加入单张图片真实框列表
                    one_image = per_image(image_name, img_path, int(
                        height), int(width), int(channels), truebox_dict_list)
                    # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
                    # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
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
                    xmin = float(min(float(one_line[2]), float(one_line[4])))
                    ymin = float(min(float(one_line[3]), float(one_line[5])))
                    xmax = float(max(float(one_line[2]), float(one_line[4])))
                    ymax = float(max(float(one_line[3]), float(one_line[5])))
                    one_image = per_image(image_name, img_path, int(
                        height), int(width), int(channels), truebox_dict_list)
                    # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
                    # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
                    total_images_data_list.append(one_image)

    print('Total: %d images extract, Done!' % len(total_images_data_list))
    if len(image_erro) != 0:
        for a in image_erro:
            print('\n%s erro!' % a)

    return total_images_data_list


def from_yolo(input_path, class_list):
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    image_path = os.path.join(input_path, 'images')   # 图片路径
    src_lab_path = os.path.join(
        input_path, 'source label')     # 对应图片的source label文件路径
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
                    '\\')[-1]).replace('.txt', '.jpg')
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
                xmin = float(min(float(bbox[0]), float(bbox[1])))
                ymin = float(min(float(bbox[2]), float(bbox[3])))
                xmax = float(max(float(bbox[1]), float(bbox[0])))
                ymax = float(max(float(bbox[3]), float(bbox[2])))
                truebox_dict_list.append(true_box(
                    cls, xmin, ymin, xmax, ymax))  # 使用动态名称变量将单个真实框加入单张图片真实框列表
        one_image = per_image(image_name, img_path, int(
                        height), int(width), int(channels), truebox_dict_list)
        # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
        # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
        total_images_data_list.append(one_image)
    print('Total: %d images extract, Done!' % len(total_images_data_list))

    return total_images_data_list

    
def from_sjt(input_path, class_list):
    
    """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

    key = 'imageName'
    image_path = os.path.join(input_path, 'images')   # 图片路径
    # 对应图片的source label文件路径
    src_lab_path = os.path.join(input_path, 'source label')
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
                xmin = int(x)
                ymin = int(y)
                xmax = int(x + one_box['w'])
                ymax = int(y + one_box['h'])
                cls = one_box['Category']
                cls = cls.replace(' ', '').lower()
                if cls == 'turningleft' or cls == 'turningright':
                    if one_box['Object_type'] == 'Traffic lights':
                        cls = 'trafficlights' + cls
                    else:
                        cls = cls
                if cls not in class_list:
                    continue
                if xmax > xmin and ymax > ymin:
                    truebox_dict_list.append(true_box(cls, xmin, ymin, xmax, ymax))  # 使用动态名称变量将单个真实框加入单张图片真实框列表
        one_image = per_image(img_name, img_path, int(
            height), int(width), int(channels), truebox_dict_list)
        # TODO
        # one_image.get_true_box_mask()   # 获取真实框在图像上的掩码图
        # 使用动态名称变量将单张图对象，添加进全数据集数据列表中
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
            xml.write('\t<folder>' + 'WH_data' + '</folder>\n')
            xml.write('\t<filename>' + image_data.image_name + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>WH Data</database>\n')
            xml.write('\t\t<annotation>WH</annotation>\n')
            xml.write('\t\t<image>flickr</image>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t</source>\n')
            xml.write('\t<owner>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t\t<name>WH</name>\n')
            xml.write('\t</owner>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(image_data.width) + '</width>\n')
            xml.write('\t\t<height>' + str(image_data.height) + '</height>\n')
            xml.write('\t\t<depth>' + str(image_data.channels) + '</depth>\n')
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
                              str(float(box.xmin)) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' +
                              str(float(box.ymin)) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' +
                              str(float(box.xmax)) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' +
                              str(float(box.ymax)) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')
            xml.write('</annotation>')
        xml_count += 1
        # print(image_data.image_path + " has saved in " + xml_path)
    print('Total: %d file change save to %s, Done!' % (xml_count, output_path))


def in_func_None(*args):
    """如无对应输入标签函数，提示用户添加输入标签函数"""
    print("\n如无对应输入标签函数，请添加输入标签函数。")
    return 0


in_func_dict = {"ldp": from_ldp, "hy": from_hy_dataset, "myxb": from_hy_dataset, "hy_highway": from_hy_highway,
                "pascal": from_pascal, "kitti": from_kitti, "coco": from_coco,
                "cctsdb": from_cctsdb, "lisa": from_lisa, "yolo": from_yolo, "hanhe": from_yolo,
                "sjt": from_sjt}
out_func_dict = {"ldp": to_pascal, "pascal": to_pascal}


def pickup_data_from_function(input_label_style, *args):
    """根据输入类别挑选转换函数"""
    # 返回对应数据获取函数
    return in_func_dict.get(input_label_style, in_func_None)(*args)


def pickup_data_out_function(output_label_style, *args):
    """根据输入类别挑选转换函数"""
    # 返回对应数据输出函数
    return out_func_dict.get(output_label_style, out_func_None)(*args)
