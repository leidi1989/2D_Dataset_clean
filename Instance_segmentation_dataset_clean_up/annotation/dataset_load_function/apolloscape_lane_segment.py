'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-27 15:19:44
LastEditors: Leidi
LastEditTime: 2021-12-31 14:38:23
'''
import cv2
import numpy as np
from collections import namedtuple

from utils.utils import *
from base.image_base import *
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_segmentation_list
from utils.convertion_function import true_segmentation_to_true_box


def load_annotation(dataset: dict, source_annotation_name: str, process_output: dict) -> None:

    Label = namedtuple('Label', [
        'name',
        'id',
        'trainId',
        'category',
        'categoryId',
        'hasInstances',
        'ignoreInEval',
        'color', ])

    labels = [
        Label('void', 0, 0, 'void', 0,
              False, False, (0, 0, 0)),
        Label('s_w_d', 200, 1, 'dividing', 1,
              False, False, (70, 130, 180)),
        Label('s_y_d', 204, 2, 'dividing', 1,
              False, False, (220, 20, 60)),
        Label('ds_w_dn', 213, 3, 'dividing', 1,
              False, True, (128, 0, 128)),
        Label('ds_y_dn', 209, 4, 'dividing',
              1, False, False, (255, 0, 0)),
        Label('sb_w_do', 206, 5, 'dividing',
              1, False, True, (0, 0, 60)),
        Label('sb_y_do', 207, 6, 'dividing',
              1, False, True, (0, 60, 100)),
        Label('b_w_g', 201, 7, 'guiding', 2,
              False, False, (0, 0, 142)),
        Label('b_y_g', 203, 8, 'guiding', 2,
              False, False, (119, 11, 32)),
        Label('db_w_g', 211, 9, 'guiding', 2,
              False, True, (244, 35, 232)),
        Label('db_y_g', 208, 10, 'guiding', 2,
              False, True, (0, 0, 160)),
        Label('db_w_s', 216, 11, 'stopping', 3,
              False, True, (153, 153, 153)),
        Label('s_w_s', 217, 12, 'stopping', 3,
              False, False, (220, 220, 0)),
        Label('ds_w_s', 215, 13, 'stopping', 3,
              False, True, (250, 170, 30)),
        Label('s_w_c', 218, 14, 'chevron', 4,
              False, True, (102, 102, 156)),
        Label('s_y_c', 219, 15, 'chevron', 4,
              False, True, (128, 0, 0)),
        Label('s_w_p', 210, 16, 'parking', 5,
              False, False, (128, 64, 128)),
        Label('s_n_p', 232, 17, 'parking', 5,
              False, True, (238, 232, 170)),
        Label('c_wy_z', 214, 18, 'zebra', 6,
              False, False, (190, 153, 153)),
        Label('a_w_u', 202, 19, 'thru/turn', 7,
              False, True, (0, 0, 230)),
        Label('a_w_t', 220, 20, 'thru/turn', 7,
              False, False, (128, 128, 0)),
        Label('a_w_tl', 221, 21, 'thru/turn', 7,
              False, False, (128, 78, 160)),
        Label('a_w_tr', 222, 22, 'thru/turn', 7,
              False, False, (150, 100, 100)),
        Label('a_w_tlr', 231, 23, 'thru/turn', 7,
              False, True, (255, 165, 0)),
        Label('a_w_l', 224, 24, 'thru/turn', 7,
              False, False, (180, 165, 180)),
        Label('a_w_r', 225, 25, 'thru/turn', 7,
              False, False, (107, 142, 35)),
        Label('a_w_lr', 226, 26, 'thru/turn', 7,
              False, False, (201, 255, 229)),
        Label('a_n_lu', 230, 27, 'thru/turn', 7,
              False, True, (0, 191, 255)),
        Label('a_w_tu', 228, 28, 'thru/turn', 7,
              False, True, (51, 255, 51)),
        Label('a_w_m', 229, 29, 'thru/turn', 7,
              False, True, (250, 128, 114)),
        Label('a_y_t', 233, 30, 'thru/turn', 7,
              False, True, (127, 255, 0)),
        Label('b_n_sr', 205, 31, 'reduction', 8,
              False, False, (255, 128, 0)),
        Label('d_wy_za', 212, 32, 'attention',
              9, False, True, (0, 255, 255)),
        Label('r_wy_np', 227, 33, 'no parking', 10,
              False, False, (178, 132, 190)),
        Label('vom_wy_n', 223, 34, 'others', 11,
              False, True, (128, 128, 64)),
        Label('om_n_n', 250, 35, 'others', 11,
              False, False, (102, 0, 204)),
        Label('noise', 249, 255, 'ignored', 255,
              False, True, (0, 153, 153)),
        Label('ignored', 255, 255, 'ignored', 255,
              False, True, (255, 255, 255)), ]

    true_segment_list = []
    # 获取data字典中images内的图片信息，file_name、height、width
    image_name = (source_annotation_name.split(
        '.')[0] + '.' + dataset['temp_image_form']).replace('_bin', '')
    image_name_new = dataset['file_prefix'] + image_name
    image_path = os.path.join(
        dataset['temp_images_folder'], image_name_new)
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # 转换png标签为IMAGE类实例
    source_annotation_path = os.path.join(
        dataset['source_annotations_folder'], source_annotation_name)
    annotation_image = cv2.imread(source_annotation_path)
    mask = np.zeros_like(annotation_image)
    for one in labels:
        if one.name == 'void':
            continue
        bgr = [one.color[2], one.color[1], one.color[0]]
        for n, color in enumerate(bgr):
            mask[:, :, n] = color
        image_xor = cv2.bitwise_xor(mask, annotation_image)
        image_gray = cv2.cvtColor(image_xor.copy(), cv2.COLOR_BGR2GRAY)
        _, thresh1 = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY_INV)
        if np.all(thresh1 == 0):
            continue
        contours, _ = cv2.findContours(
            thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for point in contours:
            point = np.squeeze(point)
            point = np.squeeze(point)
            point = point.tolist()
            if 3 > len(point):
                continue
            true_segment_list.append(TRUE_SEGMENTATION(one.name, point))

    # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
    # 并将初始化后的对象存入total_images_data_list
    image = IMAGE(image_name, image_name_new, image_path, height,
                  width, channels, [], true_segment_list)
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
