'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2022-02-22 14:10:07
'''
import os
import cv2
import json

from utils.utils import *
from base.image_base import *
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_segmentation_list
from utils.convertion_function import true_segmentation_to_true_box


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
    start_point_dist_threshhold = {'lane/crosswalk': 800,
                                   'lane/doubleother': 300,
                                   'lane/doublewhite': 300,
                                   'lane/doubleyellow': 300,
                                   'lane/roadcurb': 50,
                                   'lane/singleother': 70,
                                   'lane/singlewhite': 45,
                                   'lane/singleyellow': 70}
    dist_var_threshhold = {'lane/crosswalk': 5000,
                           'lane/doubleother': 5000,
                           'lane/doublewhite': 5000,
                           'lane/doubleyellow': 5000,
                           'lane/roadcurb': 5000,
                           'lane/singleother': 5000,
                           'lane/singlewhite': 5000,
                           'lane/singleyellow': 5000}
    one_line_expand_offset = 5

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

    # if image_name_new != 'bdd100k@00db9030-5102ed41.png':
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

    # 车道线提取，lane单双线标注分类
    object_segment_one_line_lane_list = []
    object_segment_double_line_lane_pair_list = []
    for lane in object_segment_lane_list:
        # 对贝塞尔标注线段进行解析并按min_y的x坐标进行排序
        if 2 == len(lane['poly2d']):
            segmentation_point_list = [x[0:-1] for x in lane['poly2d']]
            line_point_list = [lane['poly2d'][0][0:-1]]
            line_point_list.append(lane['poly2d'][1][0:-1])
            line_point_list.sort(key=lambda line_point: (
                line_point[1], -line_point[0]), reverse=True)
            lane.update({'line_point_list': line_point_list})
            lane.update({'zero_to_start_point_dist': dist(
                np.array(lane['line_point_list'][0]), np.array([0, 0]))})
        else:
            segmentation_point_list = [x[0:-1] for x in lane['poly2d']]
            line_point_list = [lane['poly2d'][0][0:-1]]
            for r in range(1, 21):
                r = r / 20
                line_point_list.append(calNextPoints(
                    segmentation_point_list, rate=r)[0])
            line_point_list.sort(key=lambda line_point: (
                line_point[1], -line_point[0]))
            lane.update({'line_point_list': line_point_list})
            lane.update({'zero_to_start_point_dist': dist(
                np.array(lane['line_point_list'][0]), np.array([0, 0]))})
    # 对标注线段按起始点到[0, 0]点距离进行排序
    object_segment_lane_list.sort(key=lambda x: x['zero_to_start_point_dist'])

    # 对线段进行类别划分
    lane_class_dict = {}
    for line in object_segment_lane_list:
        if line['category'] not in lane_class_dict:
            lane_class_dict.update({line['category']: [line]})
        else:
            lane_class_dict[line['category']].append(line)

    # 对进行类别划分后的车道线按单双线标注进行分类
    temp_line = {}
    for key, value in lane_class_dict.items():
        for line in value:
            if not temp_line:
                temp_line = line
            else:
                if temp_line['category'] != key:
                    object_segment_one_line_lane_list.append(temp_line)
                    temp_line = line
                else:
                    line_points_dist = []
                    for m, n in zip(temp_line['line_point_list'], line['line_point_list']):
                        line_points_dist.append(dist(np.array(m), np.array(n)))
                    line_points_dist = np.array(line_points_dist)
                    line_points_dist_var = np.var(line_points_dist)
                    line_points_dist_mean = np.mean(line_points_dist)
                    lines_start_point_dist = dist(np.array(
                        temp_line['line_point_list'][0]), np.array(line['line_point_list'][0]))
                    lines_end_point_dist = dist(np.array(
                        temp_line['line_point_list'][-1]), np.array(line['line_point_list'][-1]))
                    if (lines_start_point_dist <= start_point_dist_threshhold[key.replace(' ', '')] or
                        lines_end_point_dist <= start_point_dist_threshhold[key.replace(' ', '')]) \
                            and line_points_dist_mean <= start_point_dist_threshhold[key.replace(' ', '')]*2 \
                            and line_points_dist_var <= dist_var_threshhold[key.replace(' ', '')]:
                        object_segment_double_line_lane_pair_list.append(
                            [temp_line, line])
                        temp_line = {}
                    else:
                        object_segment_one_line_lane_list.append(temp_line)
                        temp_line = line

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
    for object in object_segment_one_line_lane_list:
        clss = object['category']
        clss = clss.replace(' ', '').lower()
        segmentation_point_list = [x[0:-1] for x in object['poly2d']]
        line_point_list = [object['poly2d'][0][0:-1]]
        # 直线
        if 2 == len(segmentation_point_list):
            line_point_list_1 = [[x - one_line_expand_offset for x in object['poly2d'][0][0:-1]],
                                 [x - one_line_expand_offset for x in object['poly2d'][1][0:-1]]]

            line_point_list_2 = [[x + one_line_expand_offset for x in object['poly2d'][0][0:-1]],
                                 [x + one_line_expand_offset for x in object['poly2d'][1][0:-1]]]

            line_point_list_l = np.array(line_point_list_1)
            line_point_list_r = np.flipud(np.array(line_point_list_2))
            line_point_list_loop = np.append(
                line_point_list_l, line_point_list_r, axis=0)
            line_point_list_loop = np.maximum(line_point_list_loop, 0)
            line_point_list_loop[:, 0] = np.minimum(
                line_point_list_loop[:, 0], 1280)
            line_point_list_loop[:, 1] = np.minimum(
                line_point_list_loop[:, 1], 720)
            line_point_list_loop = line_point_list_loop.tolist()
            one_true_segment = TRUE_SEGMENTATION(clss,
                                                 line_point_list_loop
                                                 )
            true_segment_list.append(one_true_segment)
        # 贝塞尔曲线
        else:
            # 单线左侧边缘
            line_point_list_1 = [[
                x - one_line_expand_offset for x in object['poly2d'][0][0:-1]]]
            line_point_list_1_c = []
            for points in object['poly2d']:
                line_point_list_1_c.append(
                    [points[0:-1][0] - one_line_expand_offset, points[0:-1][1]])
            for r in range(1, 21):
                r = r / 20
                line_point_list_1.append(calNextPoints(
                    line_point_list_1_c, rate=r)[0])
            # 单线右侧边缘
            line_point_list_2 = [[
                x - one_line_expand_offset for x in object['poly2d'][0][0:-1]]]
            line_point_list_2_c = []
            for points in object['poly2d']:
                line_point_list_2_c.append(
                    [points[0:-1][0] + one_line_expand_offset, points[0:-1][1]])
            for r in range(1, 21):
                r = r / 20
                line_point_list_2.append(calNextPoints(
                    line_point_list_2_c, rate=r)[0])

            line_point_list_l = np.array(line_point_list_1)
            line_point_list_r = np.flipud(np.array(line_point_list_2))
            line_point_list_loop = np.append(
                line_point_list_l, line_point_list_r, axis=0)
            line_point_list_loop = np.maximum(line_point_list_loop, 0)
            line_point_list_loop[:, 0] = np.minimum(
                line_point_list_loop[:, 0], 1280)
            line_point_list_loop[:, 1] = np.minimum(
                line_point_list_loop[:, 1], 720)
            line_point_list_loop = line_point_list_loop.tolist()

            one_true_segment = TRUE_SEGMENTATION(clss,
                                                 line_point_list_loop
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
