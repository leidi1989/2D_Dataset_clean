'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2021-11-25 17:16:27
'''
import os
import re
import cv2
import json
import operator
import functools

from utils.utils import *
from base.image_base import *
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_segmentation_list


def load_annotation(dataset: dict, source_annotation_name: str, process_output: dict) -> None:
    """[单进程读取标签]

    Args:
        dataset (dict): [数据集信息字典]
        source_annotation_path (str): [源标签路径]
        process_output (dict): [进程通信字典]
    """

    area_list = ['area/drivable',
                 'area/alternative',
                 'area/unknown'
                 ]
    pair_offset = 150
    source_annotation_path = os.path.join(
        dataset['source_annotations_folder'], source_annotation_name)
    with open(source_annotation_path, 'r') as f:
        data = json.loads(f.read())
    true_box_list = []
    true_segment_list = []
    # 获取data字典中images内的图片信息，file_name、height、width
    image_name = source_annotation_name.split(
        '.')[0] + '.' + dataset['temp_image_form']
    image_name_new = dataset['file_prefix'] + image_name

    # TODO debug
    # if os.path.splitext(image_name_new)[0] != 'bdd100k@00a2e3ca-62992459':
    #     return

    image_path = os.path.join(
        dataset['temp_images_folder'], image_name_new)
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # 标注按定义类别分类
    object_box_list = []
    object_segment_area_list = []
    object_segment_lane_list = []
    for object in data['frames'][0]['objects']:
        if 'box2d' in object:
            object_box_list.append(object)
        if 'poly2d' in object:
            clss = object['category']
            clss = clss.replace(' ', '').lower()
            if clss in area_list:
                object_segment_area_list.append(object)
            else:
                if object['attributes']['direction'] == 'vertical':
                    continue
                object_segment_lane_list.append(object)

    # true box
    for object in object_box_list:
        clss = object['category']
        clss = clss.replace(' ', '').lower()
        one_true_box = TRUE_BOX(clss,
                                object['box2d']['x1'],
                                object['box2d']['y1'],
                                object['box2d']['x2'],
                                object['box2d']['y2'],
                                color=object['attributes']['trafficLightColor'],
                                occlusion=object['attributes']['occluded']
                                )
        true_box_list.append(one_true_box)

    # object segment area
    for object in object_segment_area_list:
        clss = object['category']
        clss = clss.replace(' ', '').lower()
        segmentation_point_list = []
        last_point = ''
        temp_point = []
        c_count = 0
        # 三阶贝塞尔曲线解算
        for point in object['poly2d']:
            if point[2] == 'L':
                if '' == last_point:
                    segmentation_point_list.append(point[0:-1])
                    temp_point.append(point[0:-1])
                    last_point = 'L'
                elif 'L' == last_point:
                    segmentation_point_list += temp_point
                    temp_point = []
                    temp_point.append(point[0:-1])
                    last_point = 'L'
            else:
                temp_point.append(point[0:-1])
                last_point = 'C'
                c_count += 1
                if 3 == c_count:
                    segmentation_point_list.append(temp_point[0])
                    bezier_line = []
                    for r in range(1, 21):
                        r = r / 20
                        bezier_line.append(calNextPoints(
                            temp_point, rate=r)[0])
                    segmentation_point_list += bezier_line
                    temp_point = [temp_point[-1]]
                    last_point = 'L'
                    c_count = 0
        segmentation_point_list = np.array(segmentation_point_list)
        segmentation_point_list = np.maximum(segmentation_point_list, 0)
        segmentation_point_list[:, 0] = np.minimum(
            segmentation_point_list[:, 0], 1280)
        segmentation_point_list[:, 1] = np.minimum(
            segmentation_point_list[:, 1], 720)
        segmentation_point_list = segmentation_point_list.tolist()
        one_true_segment = TRUE_SEGMENTATION(clss,
                                             segmentation_point_list
                                             )
        true_segment_list.append(one_true_segment)

    # lane单双线标注分类
    object_segment_one_line_lane_list = []
    object_segment_double_line_lane_pair_list = []

    # 将线段按贝塞尔曲线标注点数量进行分类
    compair_dict = {}
    for line in object_segment_lane_list:
        if len(line['poly2d']) not in compair_dict:
            compair_dict.update({len(line['poly2d']): [line]})
        else:
            compair_dict[len(line['poly2d'])].append(line)

    for key, value in compair_dict.items():
        # 对贝塞尔标注线段进行解析并按min_y的x坐标进行排序
        for line in value:
            if 2 == key:
                segmentation_point_list = [x[0:-1] for x in line['poly2d']]
                line_point_list = [line['poly2d'][0][0:-1]]
                line_point_list.append(line['poly2d'][1][0:-1])
                line_point_list.sort(key=lambda line_point: (
                    line_point[1], -line_point[0]), reverse=True)
                line.update({'line_point_list': line_point_list})
            else:
                segmentation_point_list = [x[0:-1] for x in line['poly2d']]
                line_point_list = [line['poly2d'][0][0:-1]]
                for r in range(1, 21):
                    r = r / 20
                    line_point_list.append(calNextPoints(
                        segmentation_point_list, rate=r)[0])
                line_point_list.sort(key=lambda line_point: (
                    line_point[1], -line_point[0]))
                line.update({'line_point_list': line_point_list})
        # # 对标注线段按min_y的x坐标进行排序
        compair_dict[key] = sorted(
            value, key=functools.cmp_to_key(bdd100k_line_sort))

    # 将线段按单双线标注进行分类
    temp_line = {}
    for compair_key, compair_value in compair_dict.items():
        if 1 != len(compair_value):
            lane_class_dict = {}
            # 对线段进行类别划分
            for one_line in compair_value:
                if one_line['category'] not in lane_class_dict:
                    lane_class_dict.update({one_line['category']:[one_line]})
                else:
                    lane_class_dict[one_line['category']].append(one_line)
            # 对进行类别划分后的车道线按单双线标注进行分类
            for key, value in lane_class_dict.items():
                for one_line in value:
                    if not len(temp_line):
                        temp_line = one_line
                    else:
                        if len(temp_line['poly2d']) != compair_key:
                            object_segment_one_line_lane_list.append(temp_line)
                            temp_line = one_line
                        elif 4 == compair_key:
                            temp_line_start_point = np.array(temp_line['line_point_list'][0])
                            one_line_start_point = np.array(one_line['line_point_list'][0])
                            start_point_dist = dist(temp_line_start_point, one_line_start_point)
                            if start_point_dist <= pair_offset \
                                    and (temp_line['category'] == one_line['category']):
                                object_segment_double_line_lane_pair_list.append(
                                    [temp_line, one_line])
                                temp_line = {}
                            else:
                                object_segment_one_line_lane_list.append(temp_line)
                                temp_line = one_line
                        else:
                            temp_line_start_point = np.array(temp_line['line_point_list'][0])
                            one_line_start_point = np.array(one_line['line_point_list'][0])
                            start_point_dist = dist(temp_line_start_point, one_line_start_point)
                            if start_point_dist <= pair_offset \
                                    and (temp_line['category'] == one_line['category']):
                                object_segment_double_line_lane_pair_list.append(
                                    [temp_line, one_line])
                                temp_line = {}
                            else:
                                object_segment_one_line_lane_list.append(temp_line)
                                temp_line = one_line
        else:
            for one_line in value:
                object_segment_one_line_lane_list.append(one_line)

    # object segment double line lane
    for m, n in object_segment_double_line_lane_pair_list:
        clss = m['category']
        clss = clss.replace(' ', '').lower()
        # line 1
        segmentation_point_list = [x[0:-1] for x in m['poly2d']]
        line_point_list_1 = [m['poly2d'][0][0:-1]]
        for r in range(1, 21):
            r = r / 20
            line_point_list_1.append(calNextPoints(
                segmentation_point_list, rate=r)[0])
        # line 2
        segmentation_point_list = [x[0:-1] for x in n['poly2d']]
        line_point_list_2 = [n['poly2d'][0][0:-1]]
        for r in range(1, 21):
            r = r / 20
            line_point_list_2.append(calNextPoints(
                segmentation_point_list, rate=r)[0])
        pair_line_dist_0_0 = dist(
            np.array(line_point_list_1[0]), np.array(line_point_list_2[0]))
        pair_line_dist_0_1 = dist(
            np.array(line_point_list_1[0]), np.array(line_point_list_2[-1]))
        if pair_line_dist_0_0 <= pair_line_dist_0_1:
            line_point_list_2.reverse()
        line_point_list_1 += line_point_list_2
        line_point_list_1 = np.array(line_point_list_1)
        line_point_list_1 = np.maximum(line_point_list_1, 0)
        line_point_list_1[:, 0] = np.minimum(line_point_list_1[:, 0], 1280)
        line_point_list_1[:, 1] = np.minimum(line_point_list_1[:, 1], 720)
        line_point_list_1 = line_point_list_1.tolist()
        one_true_segment = TRUE_SEGMENTATION(clss,
                                             line_point_list_1
                                             )
        true_segment_list.append(one_true_segment)

    # object segment one line lane
    line_expand_offset = [3, 3]
    for object in object_segment_one_line_lane_list:
        clss = object['category']
        clss = clss.replace(' ', '').lower()
        segmentation_point_list = [x[0:-1] for x in object['poly2d']]
        line_point_list_1 = [object['poly2d'][0][0:-1]]
        for r in range(1, 21):
            r = r / 20
            line_point_list_1.append(calNextPoints(
                segmentation_point_list, rate=r)[0])
        line_point_list_l = np.array(
            line_point_list_1) - np.array(line_expand_offset)
        line_point_list_r = np.flipud(np.array(
            line_point_list_1) + np.array(line_expand_offset))
        line_point_list_1 = np.append(
            line_point_list_l, line_point_list_r, axis=0)
        line_point_list_1 = np.maximum(line_point_list_1, 0)
        line_point_list_1[:, 0] = np.minimum(line_point_list_1[:, 0], 1280)
        line_point_list_1[:, 1] = np.minimum(line_point_list_1[:, 1], 720)
        line_point_list_1 = line_point_list_1.tolist()
        one_true_segment = TRUE_SEGMENTATION(clss,
                                             line_point_list_1
                                             )
        true_segment_list.append(one_true_segment)

    # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
    # 并将初始化后的对象存入total_images_data_list
    image = IMAGE(image_name, image_name_new, image_path, height,
                  width, channels, true_box_list, true_segment_list)
    # 读取目标标注信息，输出读取的source annotation至temp annotation
    if image == None:
        return
    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        image.file_name_new + '.' + dataset['temp_annotation_form'])
    modify_true_segmentation_list(
        image, dataset['modify_class_dict'])
    if dataset['class_pixel_distance_dict'] is not None:
        class_segmentation_pixel_limit(dataset, image.true_segmentation_list)
    if 0 == len(image.true_segmentation_list) and 0 == len(image.true_box_list):
        print('{} has not true segmentation and box, delete!'.format(
            image.image_name_new))
        os.remove(image.image_path)
        process_output['no_segmentation'] += 1
        process_output['fail_count'] += 1
        return
    if TEMP_OUTPUT(temp_annotation_output_path, image):
        process_output['temp_file_name_list'].append(image.file_name_new)
        process_output['success_count'] += 1
    else:
        process_output['fail_count'] += 1
        return

    return


# def load_annotation(dataset: dict, source_annotation_name: str, process_output: dict) -> None:
#     """[单进程读取标签]

#     Args:
#         dataset (dict): [数据集信息字典]
#         source_annotation_path (str): [源标签路径]
#         process_output (dict): [进程通信字典]
#     """

#     area_list = ['area/drivable',
#                  'area/alternative',
#                  'area/unknown'
#                  ]
#     source_annotation_path = os.path.join(
#         dataset['source_annotations_folder'], source_annotation_name)
#     with open(source_annotation_path, 'r') as f:
#         data = json.loads(f.read())
#     true_box_list = []
#     true_segment_list = []
#     # 获取data字典中images内的图片信息，file_name、height、width
#     image_name = source_annotation_name.split(
#         '.')[0] + '.' + dataset['temp_image_form']
#     image_name_new = dataset['file_prefix'] + image_name
#     image_path = os.path.join(
#         dataset['temp_images_folder'], image_name_new)
#     img = cv2.imread(image_path)
#     height, width, channels = img.shape

#     # 标注按定义类别分类
#     object_box_list = []
#     object_segment_area_list = []
#     object_segment_lane_list = []
#     for object in data['frames'][0]['objects']:
#         if 'box2d' in object:
#             object_box_list.append(object)
#         if 'poly2d' in object:
#             clss = object['category']
#             clss = clss.replace(' ', '').lower()
#             if clss in area_list:
#                 object_segment_area_list.append(object)
#             else:
#                 if object['attributes']['direction'] == 'vertical':
#                     continue
#                 object_segment_lane_list.append(object)

#     # true box
#     for object in object_box_list:
#         clss = object['category']
#         clss = clss.replace(' ', '').lower()
#         one_true_box = TRUE_BOX(clss,
#                                 object['box2d']['x1'],
#                                 object['box2d']['y1'],
#                                 object['box2d']['x2'],
#                                 object['box2d']['y2'],
#                                 color=object['attributes']['trafficLightColor'],
#                                 occlusion=object['attributes']['occluded']
#                                 )
#         true_box_list.append(one_true_box)

#     # object segment area
#     for object in object_segment_area_list:
#         clss = object['category']
#         clss = clss.replace(' ', '').lower()
#         segmentation_point_list = []
#         last_point = ''
#         temp_point = []
#         c_count = 0
#         # 三阶贝塞尔曲线解算
#         for point in object['poly2d']:
#             if point[2] == 'L':
#                 if '' == last_point:
#                     segmentation_point_list.append(point[0:-1])
#                     temp_point.append(point[0:-1])
#                     last_point = 'L'
#                 elif 'L' == last_point:
#                     segmentation_point_list += temp_point
#                     temp_point = []
#                     temp_point.append(point[0:-1])
#                     last_point = 'L'
#             else:
#                 temp_point.append(point[0:-1])
#                 last_point = 'C'
#                 c_count += 1
#                 if 3 == c_count:
#                     segmentation_point_list.append(temp_point[0])
#                     bezier_line = []
#                     for r in range(1, 11):
#                         r = r / 10
#                         bezier_line.append(calNextPoints(
#                             temp_point, rate=r)[0])
#                     segmentation_point_list += bezier_line
#                     temp_point = [temp_point[-1]]
#                     last_point = 'L'
#                     c_count = 0
#         segmentation_point_list = np.array(segmentation_point_list)
#         segmentation_point_list = np.maximum(segmentation_point_list, 0)
#         segmentation_point_list[:, 0] = np.minimum(
#             segmentation_point_list[:, 0], 1280)
#         segmentation_point_list[:, 1] = np.minimum(
#             segmentation_point_list[:, 1], 720)
#         segmentation_point_list = segmentation_point_list.tolist()
#         one_true_segment = TRUE_SEGMENTATION(clss,
#                                              segmentation_point_list
#                                              )
#         true_segment_list.append(one_true_segment)


#     # lane单双线标注分类
#     pair_offset = 70
#     object_segment_one_line_lane_list = []
#     object_segment_double_line_lane_pair_list = []

#     compair_dict = {}
#     # 将线段进行分类
#     for line in object_segment_lane_list:
#         if len(line['poly2d']) not in compair_dict:
#             compair_dict.update({len(line['poly2d']): [line]})
#         else:
#             compair_dict[len(line['poly2d'])].append(line)
#     temp_line = {}
#     for key, value in compair_dict.items():
#         if 1 != len(value):
#             for one_line in value:
#                 if not len(temp_line):
#                     temp_line = one_line
#                 else:
#                     if len(temp_line['poly2d']) != key:
#                         object_segment_one_line_lane_list.append(temp_line)
#                         temp_line = one_line
#                     elif 4 == key:
#                         if abs(temp_line['poly2d'][0][0] - one_line['poly2d'][0][0]) <= pair_offset and abs(temp_line['poly2d'][3][0] - one_line['poly2d'][3][0]) <= pair_offset:
#                             object_segment_double_line_lane_pair_list.append(
#                                 [temp_line, one_line])
#                             temp_line = {}
#                         else:
#                             object_segment_one_line_lane_list.append(temp_line)
#                             temp_line = one_line
#                     else:
#                         if abs(temp_line['poly2d'][0][0] - one_line['poly2d'][0][0]) <= pair_offset and abs(temp_line['poly2d'][1][0] - one_line['poly2d'][1][0]) <= pair_offset:
#                             object_segment_double_line_lane_pair_list.append(
#                                 [temp_line, one_line])
#                             temp_line = {}
#                         else:
#                             object_segment_one_line_lane_list.append(temp_line)
#                             temp_line = one_line
#         else:
#             for one_line in value:
#                 object_segment_one_line_lane_list.append(one_line)

#     # object segment double line lane
#     for m, n in object_segment_double_line_lane_pair_list:
#         clss = m['category']
#         clss = clss.replace(' ', '').lower()
#         # line 1
#         segmentation_point_list = [x[0:-1] for x in m['poly2d']]
#         line_point_list_1 = [m['poly2d'][0][0:-1]]
#         for r in range(1, 11):
#             r = r / 10
#             line_point_list_1.append(calNextPoints(
#                 segmentation_point_list, rate=r)[0])
#         # line 2
#         segmentation_point_list = [x[0:-1] for x in n['poly2d']]
#         line_point_list_2 = [n['poly2d'][0][0:-1]]
#         for r in range(1, 11):
#             r = r / 10
#             line_point_list_2.append(calNextPoints(
#                 segmentation_point_list, rate=r)[0])
#         pair_line_dist_0_0 = dist(
#             np.array(line_point_list_1[0]), np.array(line_point_list_2[0]))
#         pair_line_dist_0_1 = dist(
#             np.array(line_point_list_1[0]), np.array(line_point_list_2[-1]))
#         if pair_line_dist_0_0 <= pair_line_dist_0_1:
#             line_point_list_2.reverse()
#         line_point_list_1 += line_point_list_2
#         line_point_list_1 = np.array(line_point_list_1)
#         line_point_list_1 = np.maximum(line_point_list_1, 0)
#         line_point_list_1[:, 0] = np.minimum(line_point_list_1[:, 0], 1280)
#         line_point_list_1[:, 1] = np.minimum(line_point_list_1[:, 1], 720)
#         line_point_list_1 = line_point_list_1.tolist()
#         one_true_segment = TRUE_SEGMENTATION(clss,
#                                              line_point_list_1
#                                              )
#         true_segment_list.append(one_true_segment)

#     # object segment one line lane
#     line_expand_offset = [3, 3]
#     for object in object_segment_one_line_lane_list:
#         clss = object['category']
#         clss = clss.replace(' ', '').lower()
#         segmentation_point_list = [x[0:-1] for x in object['poly2d']]
#         line_point_list_1 = [object['poly2d'][0][0:-1]]
#         for r in range(1, 11):
#             r = r / 10
#             line_point_list_1.append(calNextPoints(
#                 segmentation_point_list, rate=r)[0])
#         line_point_list_l = np.array(
#             line_point_list_1) - np.array(line_expand_offset)
#         line_point_list_r = np.flipud(np.array(
#             line_point_list_1) + np.array(line_expand_offset))
#         line_point_list_1 = np.append(
#             line_point_list_l, line_point_list_r, axis=0)
#         line_point_list_1 = np.maximum(line_point_list_1, 0)
#         line_point_list_1[:, 0] = np.minimum(line_point_list_1[:, 0], 1280)
#         line_point_list_1[:, 1] = np.minimum(line_point_list_1[:, 1], 720)
#         line_point_list_1 = line_point_list_1.tolist()
#         one_true_segment = TRUE_SEGMENTATION(clss,
#                                              line_point_list_1
#                                              )
#         true_segment_list.append(one_true_segment)

#     # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
#     # 并将初始化后的对象存入total_images_data_list
#     image = IMAGE(image_name, image_name_new, image_path, height,
#                   width, channels, true_box_list, true_segment_list)
#     # 读取目标标注信息，输出读取的source annotation至temp annotation
#     if image == None:
#         return
#     temp_annotation_output_path = os.path.join(
#         dataset['temp_annotations_folder'],
#         image.file_name_new + '.' + dataset['temp_annotation_form'])
#     modify_true_segmentation_list(
#         image, dataset['modify_class_dict'])
#     if dataset['class_pixel_distance_dict'] is not None:
#         class_segmentation_pixel_limit(dataset, image.true_segmentation_list)
#     if 0 == len(image.true_segmentation_list) and 0 == len(image.true_box_list):
#         print('{} has not true segmentation and box, delete!'.format(
#             image.image_name_new))
#         os.remove(image.image_path)
#         process_output['no_segmentation'] += 1
#         process_output['fail_count'] += 1
#         return
#     if TEMP_OUTPUT(temp_annotation_output_path, image):
#         process_output['temp_file_name_list'].append(image.file_name_new)
#         process_output['success_count'] += 1
#     else:
#         process_output['fail_count'] += 1
#         return

#     return
