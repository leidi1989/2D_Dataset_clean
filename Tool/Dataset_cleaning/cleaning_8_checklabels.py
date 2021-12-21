'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:02
LastEditors: Leidi
LastEditTime: 2021-12-21 16:15:34
'''
# -*- coding: utf-8 -*-
import argparse
import cv2
import os
import numpy as np
import random
from tqdm import tqdm

from utils.utils import *
from utils.convertion_function import *


def checklabels(output_path, input_label_style, masks, check):
    """检查数据集数据信息,根据源标签绘制透明真实框"""
    ImageSets_input_path = check_output_path(os.path.join(
        output_path, 'ImageSets'))     # 获取数据集ImageSets路径
    image_path = check_output_path(output_path, "JPEGImages")
    labels_path = check_output_path(output_path, "labels")
    class_path = ImageSets_input_path + os.sep + 'classes.names'     # 获取数据集类别文件路径
    class_list = get_class(class_path)
    img_tpye = check_image_type(input_label_style)
    # 不同类别的框用不同颜色区分
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(class_list))]
    # 统计各个类别的框数
    nums = [[] for _ in range(len(class_list))]

    label_path_list = os.listdir(labels_path)
    random.shuffle(label_path_list)
    pic_num = 0
    check_count = 0
    for one_label in tqdm(label_path_list):
        check_count_total = min(len(label_path_list), check)
        if check_count != check_count_total:
            label_path = os.path.join(labels_path, one_label)   # 获取标签地址
            image_name = one_label.replace('.txt', '.%s' % img_tpye)
            drow_image_path = os.path.join(image_path, os.path.splitext(one_label)[
                                        0] + '.%s' % img_tpye)     # 获取标签对应图片地址
            img = cv2.imread(drow_image_path)  # 读取对应标签图片
            a = img.shape
            image_size = []
            for a in range(3):
                image_size.append(img.shape[a])    # 返回图片的高宽通道数
            savepath = check_output_path(output_path, "show_label")
            with open(label_path, 'r') as label:    # 读取标签内bbox信息
                for one_bbox in label.readlines():  # 获取每张图片的bbox信息
                    re_one_bbox = one_bbox.strip('\n')
                    re_one_bbox = re_one_bbox.split(' ')
                    xywh = [float(re_one_bbox[1]), float(re_one_bbox[2]), float(
                        re_one_bbox[3]), float(re_one_bbox[4])]
                    # 获取xmin、xmax、ymin、ymax值
                    bbox = revers_yolo(image_size, xywh)
                    xmin = int(bbox[0])
                    xmax = int(bbox[1])
                    ymin = int(bbox[2])
                    ymax = int(bbox[3])
                    cls = int(one_bbox.split(' ')[0])  # 获取类别索引值
                    cls = class_list[cls]   # 获取类别名称
                    try:
                        nums[class_list.index(cls)].append(cls)
                        color = colors[class_list.index(cls)]
                        if masks == 0:
                            cv2.rectangle(img, (xmin, ymin),
                                        (xmax, ymax), color, 2)
                        # 绘制透明锚框
                        else:
                            zeros1 = np.zeros((img.shape), dtype=np.uint8)
                            zeros1_mask = cv2.rectangle(
                                zeros1, (xmin, ymin), (xmax, ymax), color, thickness=-1)
                            alpha = 1   # alpha 为第一张图片的透明度
                            beta = 0.5  # beta 为第二张图片的透明度
                            gamma = 0
                            # cv2.addWeighted 将原始图片与 mask 融合
                            mask_img = cv2.addWeighted(
                                img, alpha, zeros1_mask, beta, gamma)
                            img = mask_img
                        cv2.putText(img, cls, (xmin, ymin),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
                    except ValueError:
                        print(image_name + " ValueError: " +
                            str(cls) + "is not in list")

                    path = os.path.join(savepath, os.path.splitext(one_label)[0] + '.jpg')
                    cv2.imwrite(path, img)
                    pic_num += 1
            check_count += 1

    for i in nums:
        if len(i) != 0:
            print(i[0] + ':' + str(len(i)))

    with open(savepath + os.sep + 'class_count.txt', 'w') as w:
        for i in nums:
            if len(i) != 0:
                temp = i[0] + ':' + str(len(i)) + '\n'
                w.write(temp)
        w.close()

    print("\n total box: %d \n" % pic_num)
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cleaning_8_checklabels.py')
    parser.add_argument('--out', default=r'/home/leidi/Dataset/hy_highway_myxb_sjt_coco2017_7_classes_output_20210805',
                        type=str, help='output path')
    parser.add_argument('--ilstyle', '--is', dest='ilstyle', default=r'ldp',
                        type=str, help='input labels style: ldp, hy, myxb, nuscenes, \
                                                            pascal, hy_highway, coco2017, \
                                                            kitti, cctsdb, lisa, \
                                                            hanhe，yolov5_detect, yolo, \
                                                            sjt, ccpd')
    parser.add_argument('--mask', dest='mask', default=0,
                        type=int, help='transparent bounding box mask')
    parser.add_argument('--check', dest='check', default=10,
                        type=int, help='check labels')
    opt = parser.parse_args()

    output_path = check_output_path(opt.out)
    input_label_style = opt.ilstyle
    mask = opt.mask
    check = opt.check

    if check != 0:
        print('\nStart to checklabels：')
        checklabels(output_path, input_label_style, mask, check)
        print('Checklabels done!')
