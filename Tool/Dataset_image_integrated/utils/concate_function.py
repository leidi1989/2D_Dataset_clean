# -*- coding: utf-8 -*-
import argparse
import os
from utils.utils import *
from utils.extract_bbox_function import *
from utils.extract_function import *
import cv2
from tqdm import tqdm
import time


class cut_box:
    """真实框类, clss, xmin, ymin, xmax, ymax, tool"""

    def __init__(self, clss, xmin, ymin, xmax, ymax, tool='', difficult=0):
        self.clss = clss
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.tool = tool    # bbox工具
        self.difficult = difficult


class concate_image:
    """待拼接的true_box图片类
    """

    def __init__(self, img, image_style_in, image_name_in, image_path_in, height_in, width_in, channels_in, true_box_in, concat_mate_target_list_in):
        self.img = img
        self.image_style = image_style_in
        self.image_name = image_name_in    # 图片名称
        self.image_path = image_path_in    # 图片地址
        self.height = height_in    # 图片高
        self.width = width_in  # 图片宽
        self.channels = channels_in    # 图片通道数
        self.true_box = true_box_in  # 图片真实框列表
        self.concat_mate_target_list = concat_mate_target_list_in


def load_cut_image(cut_bbox_image_list):
    """[读取图片信息]

    Parameters
    ----------
    cut_bbox_image_list : [list]
        [剪切bbox图片的地址列表]

    Returns
    -------
    [list]
        [剪切图片cv2读取的详细信息，使用concate_image类描述]
    """
    total_concate_image_list = []
    print('Start extract cut images:')
    for one_cut_image in tqdm(cut_bbox_image_list):
        img = cv2.imread(one_cut_image)     # 读取cut_image图片
        image_style = os.path.splitext(one_cut_image)[-1]
        image_name = one_cut_image.split(os.sep)[-1]        # 获取图片名称
        image_path = one_cut_image      # 获取图片路径
        height = img.shape[0]       # 获取图片高
        width = img.shape[1]        # 获取图片宽
        channels = img.shape[2]     # # 获取图片通道数
        image_class = image_name.split('_')[0]      # 获取图片类别
        concat_mate_target_list = []
        total_concate_image_list.append(concate_image(
            img, image_style, image_name, image_path, height, width, channels,
            cut_box(image_class, 0, 0, width, height), concat_mate_target_list))
        # 将图片类及真实框类实例化，并添加进total_concate_image_list

    return total_concate_image_list


def load_target_label(src_set_style, src_set_input_path, target_class_list):
    """[读取concat目标数据集标签]

    Parameters
    ----------
    src_set_style : [str]
        [目标数据集类型]
    src_set_input_path : [str]
        [目标数据集路径]

    Returns
    -------
    [list]
        [目标数据集数据列表]
    """
    data_list = []
    data_list = pickup_data_from_function(src_set_style,
                                          src_set_input_path, target_class_list)     # 抽取目标数据集标签信息

    return data_list


def get_anchor_point(per_image, concate_image):
    """[依据per_image和concate_image数据，计算可拼接的锚点]

    Parameters
    ----------
    per_image : [class]
        [目标图片类]
    concate_image : [class]
        [concat图片类]


    Returns
    -------
    [class]
        [返回concat图片在目标图片上的true_box信息]
    """

    return  # true_box


def space_mate(total_concate_image_list, total_target_dateset_list):

    concat_mate_target_list = []

    for one_concat_image in total_concate_image_list:
        for one_target_image in total_target_dateset_list:

            pass
    pass


def allot_target_image(total_concate_image_list, total_target_dateset_list):

    concat_image_count = len(total_concate_image_list)      # 声明concat图片数量
    target_image_count = len(total_target_dateset_list)     # 声明target图片数量

    concat_per_target = concat_image_count / target_image_count     # 计算分配张数

    target_class_set = get_dataset_scene(
        total_target_dateset_list)     # 获取目标数据集场景元组

    if concat_per_target < 1:
        pick_target_image_count = concat_image_count

        print(0)
    else:

        print(0)
    pass
