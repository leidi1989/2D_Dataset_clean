'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:03
LastEditors: Leidi
LastEditTime: 2021-12-21 16:19:42
'''
# -*- coding: utf-8 -*-
import os
import json as js
import xml.etree.ElementTree as ET
from tqdm import tqdm
import csv
import shutil


def hy_change_filename(output_path, root, fileName, name_pre):
    """更换json文件内key的值，若更换成功返回更名计数"""

    key = 'imageName'
    if os.path.splitext(str(fileName))[-1] == 'json':
        with open(os.path.join(root, fileName), 'r+', encoding='unicode_escape') as f:
            data = js.load(f, strict=False)
        name_count = 0
        for name in data:
            # 将json文件中对应图片名的键名key更换为name_pre+key
            if key in name:
                name[key] = name_pre + name[key]
                name_count += 1
            else:
                return name_count
        # 将更改后的json文件重写为str型，并写入输出文件夹的对应文件中
        with open(os.path.join(output_path, name_pre + fileName), 'w+', encoding='unicode_escape') as json_file:
            json_str = js.dumps(data)
            json_file.write(json_str)
            json_file.close()

    return name_count


def hy_highway_change_filename(output_path, root, fileName, name_pre):
    """更换xml文件内key节点值，若更换成功返回计数1，否则为0"""

    key = 'filename'
    if os.path.splitext(str(fileName))[-1] == 'xml':
        tree = ET.parse(os.path.join(root, fileName))   # 从xml文件中读取数据
        xmlroot = tree.getroot()    # 用getroot获取根节点，根节点也是Element对象
        name = xmlroot.find(key)    # 从根节点中搜寻key命名的节点
        if name == None:    # 判断name是否存在
            return 0
        name.text = name_pre + fileName     # 修改节点名称为key的节点值
        # 按输出路径存储修改后的xml文件
        tree.write(os.path.join(output_path, name_pre + fileName))

    return 1


def pascal_change_filename(output_path, root, fileName, name_pre):
    """更换xml文件内key节点值，若更换成功返回计数1，否则为0"""

    key = 'filename'
    if os.path.splitext(str(fileName))[-1] == 'xml':
        tree = ET.parse(os.path.join(root, fileName))   # 从xml文件中读取数据
        xmlroot = tree.getroot()    # 用getroot获取根节点，根节点也是Element对象
        name = xmlroot.find(key)    # 从根节点中搜寻key命名的节点
        if name == None:    # 判断name是否存在
            return 0
        if name_pre == None:
            name.text = os.path.splitext(fileName)[0] + '.jpg'     # 修改节点名称为key的节点值
            # 按输出路径存储修改后的xml文件
            tree.write(os.path.join(output_path, fileName))
        else:
            name.text = name_pre + \
                os.path.splitext(fileName)[0] + '.jpg'     # 修改节点名称为key的节点值
            tree.write(os.path.join(output_path, name_pre + fileName))

    return 1


def kitti_2d_change_filename(output_path, root, fileName, name_pre):
    """更换txt文件名称，并移动至source label"""

    source_file = os.path.join(root, fileName)     # 源文件
    rename_file = os.path.join(
        output_path, name_pre + fileName)    # 改名后名称
    if os.path.exists(rename_file):
        os.remove(rename_file)
    shutil.copyfile(source_file, rename_file)

    return 1


def coco_change_filename(output_path, root, fileName, name_pre):
    """更换json文件名称，并移动至source label"""

    key = 'file_name'
    if os.path.splitext(str(fileName))[-1] == 'json':
        # , encoding='unicode_escape'
        with open(os.path.join(root, fileName), 'r+', encoding='unicode_escape') as f:
            data = js.loads(f)
        name_count = 0
        for one_image in tqdm(data['images']):
            # 将json文件中对应图片名的键名key更换为name_pre+key
            if key in one_image:
                one_image[key] = name_pre + one_image[key]
                name_count += 1
            else:
                return name_count
        # 将更改后的json文件重写为str型，并写入输出文件夹的对应文件中
        with open(os.path.join(output_path, name_pre + fileName), 'w+', encoding='unicode_escape') as json_file:
            json_str = js.dumps(data)
            json_file.write(json_str)
            json_file.close()

    return name_count


def cctsdb_change_filename(output_path, root, fileName, name_pre):
    """更换txt文件名称，并移动至source label"""

    key = 'file_name'
    if os.path.splitext(str(fileName))[-1] == 'txt':
        with open(os.path.join(root, fileName), 'r+') as f:
            data_all = f.readlines()
            name_count = 0
            for data in tqdm(data_all):
                data_all[data_all.index(data)] = name_pre + \
                    data.strip('\n').replace('png', 'jpg')
                name_count += 1

    with open(os.path.join(output_path, fileName), 'w+') as f_out:
        for one in data_all:
            f_out.write(one + '\n')
        f.close()

    return name_count


def lisa_change_filename(output_path, root, fileName, name_pre):
    """更换cva文件名称，并移动至source label"""

    name_count = 0
    if fileName == 'frameAnnotationsBOX.csv':
        with open(os.path.join(root, fileName), 'r+') as f:
            data_all = []       # 声明全部数据列表
            image_name_statistics = []      # 声明图片名称统计列表
            dirc_prefix = ''    # 声明新建csv文件前缀
            for one_line in f:
                one_line = one_line.replace(
                    '\n', '').split(';')     # 获取csv文件单行数据
                if one_line[0] == 'Filename':       # 若为首行表头则跳过
                    continue
                image_name = one_line[0].split('/')     # 获取图片名称
                dirc_prefix = image_name[1].split(
                    '-')      # 获取csv文件前缀名（按不同文件夹命名）
                one_line[0] = name_pre + \
                    image_name[1].replace('--', '_')      # 修改图片名称，增加前缀
                data_all.append(one_line)       # 将文件信息加入全部数据列表
                image_name_statistics.append(image_name[1])     # 对修改文件进行统计
            image_name_statistics = set(image_name_statistics)      # 清理统计列表重复项
            name_count += len(image_name_statistics)    # 计算更改图片数

        with open(os.path.join(output_path, (dirc_prefix[0] + '_' + fileName)), 'w+', encoding='utf8', newline='') as out_f:
            writer = csv.writer(out_f)
            for a in data_all:
                writer.writerow(a)

    return name_count


def yolo_change_filename(output_path, root, fileName, name_pre):
    """更换txt格式的labels名称"""

    source_file = os.path.join(root, fileName)     # 源文件
    if name_pre == None:
        name_pre = ''
    rename_file = os.path.join(
        output_path, name_pre + fileName)    # 改名后名称
    if os.path.exists(rename_file):
        os.remove(rename_file)
    shutil.copyfile(source_file, rename_file)

    return 1


def sjt_change_filename(output_path, root, fileName, name_pre):
    """更换sjt的labels中图片文件名称"""

    source_file = os.path.join(root, fileName)     # 源文件
    rename_file = os.path.join(
        output_path, name_pre + fileName)    # 改名后名称
    if os.path.exists(rename_file):
        os.remove(rename_file)
    shutil.copyfile(source_file, rename_file)

    return 1


def nuscenes_change_filename(output_path, root, fileName, name_pre):
    """更换nuscenes标签转换为json文件后的标签中的图片名称"""
    key = 'filename'
    if os.path.splitext(str(fileName))[-1] == 'json':
        # , encoding='unicode_escape'
        with open(os.path.join(root, fileName), 'r+', encoding='utf8') as f:
            data = js.load(f)
        name_count = 0
        for one_image_true_box in tqdm(data):
            # 将json文件中对应图片名的键名key更换为name_pre+key
            if key in one_image_true_box:
                one_image_true_box_filename = one_image_true_box[key]
                one_image_true_box_filename_spilt_list = one_image_true_box_filename.split(
                    '/')
                if None != name_pre:
                    one_image_true_box_filename_spilt_list_refilename = name_pre + \
                        one_image_true_box_filename_spilt_list[-1]
                    one_image_true_box[key] = one_image_true_box_filename_spilt_list_refilename[0] + \
                        "/" + one_image_true_box_filename_spilt_list_refilename[1] + \
                        "/" + \
                        one_image_true_box_filename_spilt_list_refilename[2]
                    name_count += 1
            else:
                continue
        # 将更改后的json文件重写为str型，并写入输出文件夹的对应文件中
        with open(os.path.join(output_path, fileName), 'w+', encoding='utf8') as json_file:
            # json_str = js.dumps(data)
            # json_file.write(json_str)
            js.dump(data, json_file)
            json_file.close()

    return name_count


def ccpd_change_filename(output_path, root, fileName, name_pre):
    """[将ccpd数据集中图片名label转换为txt文件保存]

    Args:
        output_path ([str]): [输出txt文件路径]
        root ([str]): [输入图片路径]
        fileName ([str]): [读取文件名]
        name_pre ([str]): [更名前缀]

    Returns:
        [int]: [返回计数1]
    """
    label_list = []
    label_list.append(name_pre + fileName)
    with open(os.path.join(output_path, 'total_label.txt'), 'a', encoding='utf8') as out_f:
        for a in label_list:
            a = a.replace('&', '#')
            out_f.write(a+'\n')

    return 1


def func_None(*args):
    """如无对应函数，提示用户添加修改名称函数"""

    print("\nCannot find function, you shoule appen the function.")
    return 0


# 建立函数字典，供更名主函数查询
key_func_dict = {"ldp": pascal_change_filename,
                 "hy": hy_change_filename, "myxb": hy_change_filename,
                 "hy_highway": hy_highway_change_filename,
                 "voc": hy_highway_change_filename, "kitti": kitti_2d_change_filename,
                 "pascal": pascal_change_filename, "coco": coco_change_filename,
                 "cctsdb": cctsdb_change_filename, "lisa": lisa_change_filename,
                 "yolo": yolo_change_filename, "hanhe": yolo_change_filename,
                 "licenseplate": yolo_change_filename, "sjt": sjt_change_filename, 
                 "nuscenes": nuscenes_change_filename,"yolov5_detect": yolo_change_filename, 
                 "ccpd": ccpd_change_filename}


def pickup_move_function(label_type, *args):
    """根据输入类别挑选更换图片文件名"""

    return key_func_dict.get(label_type, func_None)(*args)  # 返回对应类别更名函数
