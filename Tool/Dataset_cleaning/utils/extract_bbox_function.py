# -*- coding: utf-8 -*-
import os
import cv2
from tqdm import tqdm

from utils.utils import *


class cut_image:
    """裁切的bbox图片类
    """

    def __init__(self, src_image_size, image_name_in, image_path_in, height_in, width_in, channels_in, true_box_list_in):
        self.src_image_size = src_image_size
        self.image_name = image_name_in    # 图片名称
        self.image_path = image_path_in    # 图片地址
        self.height = height_in    # 图片高
        self.width = width_in  # 图片宽
        self.channels = channels_in    # 图片通道数
        self.true_box_list = true_box_list_in  # 图片真实框列表

    def true_box_list_updata(self, one_bbox_data):
        """为per_image对象true_box_list成员添加元素"""
        self.true_box_list.append(one_bbox_data)


def creat_bbox_image(data_list, output_image_style, output_image_path, pick_classes_list):
    """[创建源图片bbox剪切图片信息]

    Parameters
    ----------
    data_list : [list]
        [数据列表]
    putout_image_style : [str]
        [输出图片类型]
    output_image_path : [str]
        [输出图片路径]
    pick_classes_list : [list]
        [挑选bbox类别列表]

    Returns
    -------
    [list]
        [输出剪切图片列表]
    """

    cut_image_list = []

    print('开始检出真实框图片')
    for one_image in tqdm(data_list):     # 遍历全部annotation数据
        img = cv2.imread(one_image.image_path,
                         cv2.IMREAD_COLOR)      # 读取one_image图片
        src_image_size = img.shape
        for i, one_bbox in enumerate(one_image.true_box_list):    # 遍历图片包含的bbox信息
            if one_bbox.clss not in pick_classes_list:      # 若bbox不在指定的提取类别列表内，则略过
                continue
            cut_box_image = img[
                int(one_bbox.ymin):int(one_bbox.ymax),
                int(one_bbox.xmin):int(one_bbox.xmax)]       # 读取bbox的ymin，ymax，xmin，xmax
            cut_box_image_shape = cut_box_image.shape
            cut_box_image_name = '{0}_{1}_from_{2}'.format(
                one_bbox.clss, i, one_image.image_name)      # 声明图片名称
            cut_image_output_path = check_out_file_exists(os.path.join(
                output_image_path, cut_box_image_name))      # 声明剪切后图片保存路径
            cut_image_list.append(
                cut_image(src_image_size,
                          cut_box_image_name,
                          cut_image_output_path,
                          cut_box_image_shape[1],
                          cut_box_image_shape[0],
                          cut_box_image_shape[2],
                          cut_box_image)
            )       # 将剪切后图片信息存入cut_image_list中

    return cut_image_list


def save_cut_bbox_image(cut_image_list, output_image_path):
    """[存储提取的bbox图片]

    Parameters
    ----------
    cut_image_list : [list]
        [提取的bbox图片nparray数据]
    output_image_path : [str]
        [输出存储bbox图片的路径]

    Returns
    -------
    [int]
        [输出存储的bbox图片数量]
    """

    n = 0
    # 遍历全部cut_image_list，将bbox图片保存为单一图片
    for one_cut_image in tqdm(cut_image_list):
        cut_image_output_path = os.path.join(
            output_image_path, one_cut_image.image_name)
        cut_box_image = one_cut_image.true_box_list
        cv2.imwrite(cut_image_output_path, cut_box_image)     # 将剪切的图片保存
        n += 1

    return n
