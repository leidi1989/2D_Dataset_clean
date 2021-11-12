# -*- coding: utf-8 -*-
import numpy as np


def yolo(size, box):
    """将坐标转换为YOLO格式，其中size为图片大小"""
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

    assert (x>=0 and w>=0 and y>=0 and h>=0), 'images bbox erro!'

    return (x, y, w, h)


def revers_yolo(size, xywh):
    """将YOLO中心点格式bbox转换为普通xmin、xmax、ymin、ymax"""

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


def coco_voc(xywh):
    """将coco中心点格式bbox转换为普通xmin、xmax、ymin、ymax"""

    x = xywh[0]
    y = xywh[1]
    w = xywh[2]
    h = xywh[3]
    bbox = []
    bbox.append(int(xywh[0]))    # xmin
    bbox.append(int(xywh[0] + xywh[2]))    # xmax
    bbox.append(int(xywh[1]))    # ymin
    bbox.append(int(xywh[1] + xywh[3]))    # ymax

    return bbox
