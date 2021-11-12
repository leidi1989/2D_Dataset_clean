'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:02
LastEditors: Leidi
LastEditTime: 2021-05-15 17:55:29
'''
import os
import shutil
import time
import argparse
from tqdm import tqdm

from utils.utils import *


def change_datast_floder(output_path, src_set_style, out_set_style, new_output_path, for_detect):
    """[将指定组织格式数据集转换为指定类型数据集组织格式]

    Parameters
    ----------
    output_path : [str]
        [数据集输入路径]
    src_set_style : [str]
        [源数据集类别]
    out_set_style : [str]
        [输出数据集类别]
    new_output_path : [str]
        [新输出路径]
    """

    if for_detect == 0:
        train_images_list_path = os.path.join(
            output_path, 'ImageSets', 'train.txt')
        val_images_list_path = os.path.join(output_path, 'ImageSets', 'test.txt')
        train_labels_output_path = check_output_path(
            os.path.join(new_output_path, 'labels', 'train'))
        val_labels_output_path = check_output_path(
            os.path.join(new_output_path, 'labels', 'val'))
        train_images_output_path = check_output_path(
            os.path.join(new_output_path, 'JPEGImages', 'train'))
        val_images_output_path = check_output_path(
            os.path.join(new_output_path, 'JPEGImages', 'val'))
        label_stype = check_src_lab_type(out_set_style)
        image_stype = check_image_type(out_set_style)
        train_labels_list = []
        val_labels_list = []

        with open(train_images_list_path, 'r') as train_images_list_path_file:    # 获取训练集标签路径，并移动训练集图片至指定输出数据集
            print('\nOutput train images to: ', train_images_output_path)
            for one in tqdm(train_images_list_path_file.readlines()):
                one = one.replace('\n', '')
                shutil.copyfile(one, check_out_file_exists(
                    os.path.join(train_images_output_path, one.split('\\')[-1])))
                train_labels_list.append(one.replace(
                    '.'+image_stype, '.'+label_stype).replace('JPEGImages', 'labels'))
            train_images_list_path_file.close()

        with open(val_images_list_path, 'r') as val_images_list_path_file:    # 获取验证集标签路径，并移动验证集图片至指定输出数据集
            print('\nOutput val images to: ', val_images_output_path)
            for one in tqdm(val_images_list_path_file.readlines()):
                one = one.replace('\n', '')
                shutil.copyfile(one, check_out_file_exists(
                    os.path.join(val_images_output_path, one.split('\\')[-1])))
                val_labels_list.append(one.replace(
                    '.'+image_stype, '.'+label_stype).replace('JPEGImages', 'labels'))
            val_images_list_path_file.close()

        print('\nOutput train labels to: ', train_labels_output_path)
        for one in tqdm(train_labels_list):     # 复制训练集标签至指定输出数据集
            one = one.replace('png','txt').replace('jpg','txt')
            new = one.split('\\')[-1]
            shutil.copyfile(one, check_out_file_exists(
                os.path.join(train_labels_output_path, new)))

        print('\nOutput val labels to: ', val_labels_output_path)
        for one in tqdm(val_labels_list):   # 复制验证集标签至指定输出数据集
            one = one.replace('png','txt').replace('jpg','txt')
            new = one.split('\\')[-1]
            shutil.copyfile(one, check_out_file_exists(
                os.path.join(val_labels_output_path, new)))
    else:
        val_images_list_path = os.path.join(output_path, 'ImageSets', 'val.txt')
        val_labels_output_path = check_output_path(
            os.path.join(new_output_path, 'labels', 'val'))
        val_images_output_path = check_output_path(
            os.path.join(new_output_path, 'JPEGImages', 'val'))
        label_stype = check_src_lab_type(out_set_style)
        image_stype = check_image_type(out_set_style)
        val_labels_list = []

        with open(val_images_list_path, 'r') as val_images_list_path_file:    # 获取验证集标签路径，并移动验证集图片至指定输出数据集
            print('\nOutput val images to: ', val_images_output_path)
            for one in tqdm(val_images_list_path_file.readlines()):
                one = one.replace('\n', '')
                shutil.copyfile(one, check_out_file_exists(
                    os.path.join(val_images_output_path, one.split('\\')[-1])))
                val_labels_list.append(one.replace(
                    '.'+image_stype, '.'+label_stype).replace('JPEGImages', 'labels'))
            val_images_list_path_file.close()

        print('\nOutput val labels to: ', val_labels_output_path)
        for one in tqdm(val_labels_list):   # 复制验证集标签至指定输出数据集
            one = one.replace('png','txt').replace('jpg','txt')
            new = one.split('\\')[-1]
            shutil.copyfile(one, check_out_file_exists(
                os.path.join(val_labels_output_path, new)))


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='tool_move.py')
    parser.add_argument('--out', default=r'/media/leidi/My Passport/data/hy_dataset/CCTSDB_analyze',
                        type=str, help='output path')
    parser.add_argument('--nout', default=r'/media/leidi/My Passport/data/hy_dataset/CCTSDB_analyze_fordetect',
                        type=str, help='new output path')
    parser.add_argument('--istyle', '--is', dest='istyle', default=r'ldp',
                        type=str, help='source detaset style: ldp')
    parser.add_argument('--omod', '--od', dest='omod', default=r'yolov5',
                        type=str, help='output detaset style: yolov5')
    parser.add_argument('--detc', '--d', dest='detc', default=1,
                        type=str, help='output detaset style: yolov5')
    opt = parser.parse_args()

    output_path = check_output_path(opt.out)
    new_output_path = check_output_path(opt.nout)
    src_set_style = opt.istyle
    out_set_style = opt.omod
    for_detect = opt.detc

    print('Start to move dataset image and label:')
    change_datast_floder(output_path, src_set_style,
                         out_set_style, new_output_path, for_detect)
    print('Move dataset image and label done!')
