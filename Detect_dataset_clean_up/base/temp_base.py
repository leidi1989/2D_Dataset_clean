'''
Description:
Version:
Author: Leidi
Date: 2021-08-03 22:18:39
LastEditors: Leidi
LastEditTime: 2021-10-28 10:43:52
'''
import os
import cv2
import numpy as np
from tqdm import tqdm
from random import random


from base.image_base import *
from annotation.annotation_temp import TEMP_LOAD, TEMP_OUTPUT
from annotation.annotation_load import annotation_load_function


def temp(dataset: dict) -> None:
    """[将源数据集标签转换为暂存数据集格式标签]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Start to transform source annotation to temp annotation:')
    annotation_load_function(
        dataset['source_dataset_stype'], dataset)
    if 0 != dataset['blurlever']:
        image_blur(dataset)
    if 0 != dataset['perspectivelever']:
        perspective_transform(dataset)


def image_blur(dataset: dict) -> None:
    """[图片模糊化]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('\nStart image blur:')
    for n in tqdm(dataset['temp_file_name_list']):
        image_path = os.path.join(
            dataset['temp_images_folder'], n + '.' + dataset['temp_image_form'])
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        # 首轮模糊
        image_blur = cv2.GaussianBlur(
            image, (dataset['blurlever']*2+1, dataset['blurlever']*2+1), dataset['blurlever']*3)
        image_blur = cv2.medianBlur(image_blur, dataset['blurlever']*2+1)
        # 分辨率模糊
        image_blur = cv2.resize(image_blur, (int(
            width*(1-0.1*dataset['blurlever'])), int(height*(1-0.1*dataset['blurlever']))), interpolation=cv2.INTER_LANCZOS4)
        image_blur = cv2.resize(
            image_blur, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        # 马赛克
        bais = dataset['blurlever']
        for m in range(height-bais):
            for n in range(width-bais):
                if m % bais == 0 and n % bais == 0:
                    for i in range(bais):
                        for j in range(bais):
                            b, g, r = image_blur[m, n]
                            image_blur[m+i, n+j] = (b, g, r)
        # 再模糊
        try:
            image_blur = cv2.GaussianBlur(
                image_blur, (dataset['blurlever'], dataset['blurlever']), dataset['blurlever'])
        except:
            image_blur = cv2.GaussianBlur(
                image_blur, (dataset['blurlever']+1, dataset['blurlever']+1), dataset['blurlever'])
        try:
            image_blur = cv2.medianBlur(image_blur, dataset['blurlever'])
        except:
            image_blur = cv2.medianBlur(image_blur, dataset['blurlever']+1)
        # 保存图片
        cv2.imwrite(image_path, image_blur)

    return


def perspective_transform(dataset: dict) -> None:
    """[将训练集图片进行透视变换]

    Args:
        dataset (dict): [数据集信息字典]
    """

    baie_scale = dataset['perspectivelever'] * 5
    print('\nStart image perspective transform:')
    for n in tqdm(dataset['temp_file_name_list']):
        image_path = os.path.join(
            dataset['temp_images_folder'],  n + '.' + dataset['temp_image_form'])
        annotaion_path = os.path.join(
            dataset['temp_annotations_folder'],  n + '.' + dataset['temp_annotation_form'])

        # 图片透视变换
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        img_space = (width*3, height*3)
        offset = [width, height]
        baies = []
        for _ in range(8):
            b = random()
            baies.append(int(b * baie_scale))
        pts1 = np.float32(
            [[0, 0], [0, width], [height, 0], [height, width]])
        pts2 = np.float32([[offset[0] + pts1[0][0] + baies[0], offset[1] + pts1[0][1] + baies[1]],
                           [offset[0] + pts1[1][0] + baies[2],
                               offset[1] + pts1[1][1] + baies[3]],
                           [offset[0] + pts1[2][0] + baies[4],
                               offset[1] + pts1[2][1] + baies[5]],
                           [offset[0] + pts1[3][0] + baies[6], offset[1] + pts1[3][1] + baies[7]]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_transform_img = cv2.warpPerspective(
            img, M, img_space, 1)

        # pts_target_min = []
        # pts_target_max = []
        # pts_target_min.append(
        #     min(pts2[0][0], pts2[2][0]))
        # pts_target_min.append(
        #     min(pts2[0][1], pts2[1][1]))
        # pts_target_max.append(
        #     max(pts2[1][0], pts2[3][0]))
        # pts_target_max.append(
        #     max(pts2[2][1], pts2[3][1]))

        # perspective_transform_img = perspective_transform_img[int(pts_target_min[1]):int(pts_target_max[1]),
        #                                                       int(pts_target_min[0]):int(pts_target_max[0]),
        #                                                       :]

        # temp annotation真实框透视变换
        image = TEMP_LOAD(dataset, annotaion_path)
        true_box_list = []
        for true_box in image.true_box_list:
            height = true_box.ymax - true_box.ymin
            width = true_box.xmax - true_box.xmin
            p0 = np.array([true_box.xmin, true_box.ymin])
            p1 = np.array([true_box.xmin + width, true_box.ymin])
            p2 = np.array([true_box.xmin, true_box.ymin + height])
            p3 = np.array([true_box.xmax, true_box.ymax])

            p0_t = []
            p1_t = []
            p2_t = []
            p3_t = []
            p0_t.append((M[0][0]*p0[0] + M[0][1]*p0[1] + M[0][2]) /
                        (M[2][0]*p0[0] + M[2][1]*p0[1] + M[2][2]))
            p0_t.append((M[1][0]*p0[0] + M[1][1]*p0[1] + M[1][2]) /
                        (M[2][0]*p0[0] + M[2][1]*p0[1] + M[2][2]))
            p1_t.append((M[0][0]*p1[0] + M[0][1]*p1[1] + M[0][2]) /
                        (M[2][0]*p1[0] + M[2][1]*p1[1] + M[2][2]))
            p1_t.append((M[1][0]*p1[0] + M[1][1]*p1[1] + M[1][2]) /
                        (M[2][0]*p1[0] + M[2][1]*p1[1] + M[2][2]))
            p2_t.append((M[0][0]*p2[0] + M[0][1]*p2[1] + M[0][2]) /
                        (M[2][0]*p2[0] + M[2][1]*p2[1] + M[2][2]))
            p2_t.append((M[1][0]*p2[0] + M[1][1]*p2[1] + M[1][2]) /
                        (M[2][0]*p2[0] + M[2][1]*p2[1] + M[2][2]))
            p3_t.append((M[0][0]*p3[0] + M[0][1]*p3[1] + M[0][2]) /
                        (M[2][0]*p3[0] + M[2][1]*p3[1] + M[2][2]))
            p3_t.append((M[1][0]*p3[0] + M[1][1]*p3[1] + M[1][2]) /
                        (M[2][0]*p3[0] + M[2][1]*p3[1] + M[2][2]))

            # true_box.xmin = min(p0_t[0], p2_t[0]) - int(pts_target_min[0])
            # true_box.ymin = min(p0_t[1], p1_t[1]) - int(pts_target_min[1])
            # true_box.xmax = max(p3_t[0], p1_t[0]) - int(pts_target_min[0])
            # true_box.ymax = max(p3_t[1], p2_t[1]) - int(pts_target_min[1])
            # true_box_list.append(true_box)
            true_box.xmin = int(min(p0_t[0], p2_t[0]))
            true_box.ymin = int(min(p0_t[1], p1_t[1]))
            true_box.xmax = int(max(p3_t[0], p1_t[0]))
            true_box.ymax = int(max(p3_t[1], p2_t[1]))
            true_box_list.append(true_box)
        height, width, _ = perspective_transform_img.shape
        image.height = height
        image.width = width
        image.true_box_list = true_box_list

        TEMP_OUTPUT(dataset, annotaion_path, image)
        cv2.imwrite(image_path, perspective_transform_img)

    return
