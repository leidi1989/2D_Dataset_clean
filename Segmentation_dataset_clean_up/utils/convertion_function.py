'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-06 09:06:35
LastEditors: Leidi
LastEditTime: 2021-12-27 17:09:15
'''
# -*- coding: utf-8 -*-
import numpy as np
from base.image_base import *


def yolo(size: list, box: list) -> tuple:
    """[将坐标转换为YOLO格式，其中size为图片大小]

    Args:
        size (list): [图片大小]
        box (list): [普通xmin、xmax、ymin、ymax]

    Returns:
        tuple: [YOLO中心点格式bbox]
    """

    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    assert (x >= 0 and w >= 0 and y >= 0 and h >= 0), 'images bbox erro!'

    return (x, y, w, h)


def revers_yolo(size: list, xywh: list) -> list:
    """[将YOLO中心点格式bbox转换为普通xmin、xmax、ymin、ymax，其中size为图片大小]

    Args:
        size (list): [图片大小]
        xywh (list): [中心店及宽高比例]

    Returns:
        list: [普通xmin、xmax、ymin、ymax列表]
    """

    image_h = size[0]
    image_w = size[1]
    x = float(xywh[0])
    y = float(xywh[1])
    w = float(xywh[2])
    h = float(xywh[3])
    bbox = []
    bbox.append(int((2*x-w)/2*image_w))
    bbox.append(int((2*x+w)/2*image_w))
    bbox.append(int((2*y-h)/2*image_h))
    bbox.append(int((2*y+h)/2*image_h))

    return bbox


def coco_voc(xywh: list) -> list:
    """[将coco中心点格式bbox转换为普通xmin、xmax、ymin、ymax]

    Args:
        xywh (list): [coco数据集坐标xywh列表]

    Returns:
        list: [普通xmin、xmax、ymin、ymax列表]
    """

    bbox = []
    bbox.append(int(xywh[0]))    # xmin
    bbox.append(int(xywh[0] + xywh[2]))    # xmax
    bbox.append(int(xywh[1]))    # ymin
    bbox.append(int(xywh[1] + xywh[3]))    # ymax

    return bbox


def true_segmentation_to_true_box(true_segmentation: TRUE_SEGMENTATION) -> list:
    """[将分割按最外围矩形框转换为bbox]

    Args:
        true_segmentation (TRUE_SEGMENTATION): [真实分割]

    Returns:
        list: [转换后真实框左上点坐标、宽、高]
    """

    segmentation = np.asarray(true_segmentation.segmentation)
    min_x = np.min(segmentation[:, 0])
    min_y = np.min(segmentation[:, 1])
    max_x = np.max(segmentation[:, 0])
    max_y = np.max(segmentation[:, 1])
    width = max_x - min_x
    hight = max_y - min_y
    bbox = [int(min_x), int(min_y), int(width), int(hight)]

    return bbox
