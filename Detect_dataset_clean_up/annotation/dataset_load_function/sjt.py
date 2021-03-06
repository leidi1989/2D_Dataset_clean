'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2022-02-17 16:03:44
'''
import os
import cv2
import json

from base.image_base import *
from utils.utils import class_pixel_limit
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_box_list


def load_annotation(dataset: dict, source_annotation_name: str, process_output,
                    change_Occlusion, change_traffic_light) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        source_annotation_path (str): [源标签路径]
        process_output ([dict]): [进程通信字典]
    """

    source_annotation_path = os.path.join(
        dataset['source_annotations_folder'], source_annotation_name)
    with open(source_annotation_path, 'r', encoding='unicode_escape') as f:
        data = json.load(f)
        
    image_name = os.path.splitext(source_annotation_name)[0] + '.jpg'
    image_name_new = dataset['file_prefix'] + image_name
    image_path = os.path.join(
        dataset['temp_images_folder'], image_name_new)
    img = cv2.imread(image_path)
    if img is None:
        print('Can not load: {}'.format(image_name_new))
        return
    height, width, channels = img.shape     # 读取每张图片的shape
    true_box_dict_list = []  # 声明每张图片真实框列表
    if len(data['boxs']):
        for one_box in data['boxs']:
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
                true_box_dict_list.append(
                    TRUE_BOX(cls, xmin, ymin, xmax, ymax, true_box_color,
                             0, occlusion=ture_box_occlusion, distance=ture_box_distance))
            else:
                print('\nBbox error!')
                continue
    true_box_dict_list = change_traffic_light(
        dataset, true_box_dict_list)  # 添加修改信号灯框名称后的真实框
    for one_true_box in true_box_dict_list:
        if one_true_box.clss not in dataset['source_class_list']:
            true_box_dict_list.pop(true_box_dict_list.index(one_true_box))
    image = IMAGE(image_name, image_name_new, image_path, int(
        height), int(width), int(channels), true_box_dict_list)

    # 将单张图对象添加进全数据集数据列表中
    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        image.file_name_new + '.' + dataset['temp_annotation_form'])
    modify_true_box_list(image, dataset['modify_class_dict'])
    if dataset['class_pixel_distance_dict'] is not None:
        class_pixel_limit(dataset, image.true_box_list)
    if 0 == len(image.true_box_list):
        print('{} has not true box, delete!'.format(image.image_name_new))
        os.remove(image.image_path)
        process_output['no_true_box_count'] += 1
        process_output['fail_count'] += 1
        return
    if TEMP_OUTPUT(temp_annotation_output_path, image):
        process_output['temp_file_name_list'].append(image.file_name_new)
        process_output['success_count'] += 1
    else:
        process_output['fail_count'] += 1
        return

    return


def change_traffic_light(dataset: dict, true_box_dict_list: list) -> list:
    """[修改数据堂信号灯标签信息，将灯与信号灯框结合]

    Args:
        true_box_dict_list (list): [源真实框]

    Returns:
        list: [修改后真实框]
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
    new_true_box_dict_list = []  # 声明新真实框列表
    for one_true_box in true_box_dict_list:  # 遍历源真实框列表
        if one_true_box.clss == 'trafficlightframe':    # 搜索trafficlightframe真实框
            if one_true_box.color == 'no':
                for light_true_box in true_box_dict_list:    # 遍历源真实框列表
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
                for light_true_box in true_box_dict_list:    # 遍历源真实框列表
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
        if one_true_box.clss in dataset['source_class_list']:
            new_true_box_dict_list.append(one_true_box)

    return new_true_box_dict_list


def change_Occlusion(source_occlusion: str) -> int:
    """[转换真实框遮挡信息]

    Args:
        source_occlusion (str): [ture box遮挡信息]

    Returns:
        int: [返回遮挡值]
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