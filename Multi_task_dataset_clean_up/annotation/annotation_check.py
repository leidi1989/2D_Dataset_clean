'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:43:21
LastEditors: Leidi
LastEditTime: 2021-12-27 17:49:40
'''
import os
import cv2
import json

from utils.utils import *
from base.image_base import *


def bbd100k(dataset: dict) -> list:
    """[读取BBD100K数据集图片类检测列表]

    Args:
        dataset (dict): [数据集信息字典]

    Returns:
        list: [数据集图片类检测列表]
    """

    check_images_list = []
    dataset['total_file_name_path'] = os.path.join(
        dataset['temp_informations_folder'], 'total_file_name.txt')
    dataset['check_file_name_list'] = annotations_path_list(
        dataset['total_file_name_path'], dataset['target_annotation_check_count'])
    for n in tqdm(dataset['check_file_name_list']):
        target_annotation_path = os.path.join(
            dataset['target_annotations_folder'],
            n + '.' + dataset['target_annotation_form'])
        with open(target_annotation_path, 'r') as f:
            data = json.loads(f.read())
            image_name = n + '.' + dataset['target_image_form']
            image_path = os.path.join(
                dataset['temp_images_folder'], image_name)
            image_size = cv2.imread(image_path).shape
            height = image_size[0]
            width = image_size[1]
            channels = image_size[2]
            true_segmentation_list = []
            true_box_list = []
            for obj in data['frames'][0]['objects']:
                if 'box2d' in obj:
                    cls = str(obj['category'])
                    cls = cls.replace(' ', '').lower()
                    if cls not in dataset['detect_class_list_new']:
                        continue
                    true_box_list.append(TRUE_BOX(cls,
                                                  obj['box2d']['x1'],
                                                  obj['box2d']['y1'],
                                                  obj['box2d']['x2'],
                                                  obj['box2d']['y2']))  # 将单个真实框加入单张图片真实框列表
                if 'poly2d' in obj:
                    cls = str(obj['category'])
                    cls = cls.replace(' ', '').lower()
                    if cls not in dataset['segment_class_list_new']:
                        continue
                    segment = []
                    for seg in obj['poly2d']:
                        segment.append(list(map(int, seg)))
                    true_segmentation_list.append(TRUE_SEGMENTATION(
                        cls, segment))  # 将单个真实框加入单张图片真实框列表
            one_image = IMAGE(image_name, image_name, image_path, int(
                height), int(width), int(channels), true_box_list, true_segmentation_list)
            f.close()
            check_images_list.append(one_image)

    return check_images_list


def yolop(dataset: dict) -> list:
    """[读取BBD100K数据集图片类检测列表]

    Args:
        dataset (dict): [数据集信息字典]

    Returns:
        list: [数据集图片类检测列表]
    """

    check_images_list = []
    dataset['total_file_name_path'] = os.path.join(
        dataset['temp_informations_folder'], 'total_file_name.txt')
    dataset['check_file_name_list'] = annotations_path_list(
        dataset['total_file_name_path'], dataset['target_annotation_check_count'])
    for n in tqdm(dataset['check_file_name_list']):
        target_annotation_path = os.path.join(
            dataset['target_annotations_folder'],
            n + '.' + dataset['target_annotation_form'])
        with open(target_annotation_path, 'r') as f:
            data = json.loads(f.read())
            image_name = n + '.' + dataset['target_image_form']
            image_path = os.path.join(
                dataset['temp_images_folder'], image_name)
            image_size = cv2.imread(image_path).shape
            height = image_size[0]
            width = image_size[1]
            channels = image_size[2]
            true_segmentation_list = []
            true_box_list = []
            for obj in data['frames'][0]['objects']:
                if 'box2d' in obj:
                    cls = str(obj['category'])
                    cls = cls.replace(' ', '').lower()
                    if cls not in dataset['detect_class_list_new']:
                        continue
                    true_box_list.append(TRUE_BOX(cls,
                                                  obj['box2d']['x1'],
                                                  obj['box2d']['y1'],
                                                  obj['box2d']['x2'],
                                                  obj['box2d']['y2']))  # 将单个真实框加入单张图片真实框列表
                if 'poly2d' in obj:
                    cls = str(obj['category'])
                    cls = cls.replace(' ', '').lower()
                    if cls not in dataset['segment_class_list_new']:
                        continue
                    segment = []
                    for seg in obj['poly2d']:
                        segment.append(list(map(int, seg)))
                    true_segmentation_list.append(TRUE_SEGMENTATION(
                        cls, segment))  # 将单个真实框加入单张图片真实框列表
            one_image = IMAGE(image_name, image_name, image_path, int(
                height), int(width), int(channels), true_box_list, true_segmentation_list)
            f.close()
            check_images_list.append(one_image)

    return check_images_list


def coco2017(dataset: dict) -> list:
    """[读取COCO2017数据集图片类检测列表]

    Args:
        dataset (dict): [数据集信息字典]

    Returns:
        list: [数据集图片类检测列表]
    """

    check_images_list = []
    dataset['check_file_name_list'] = os.listdir(
        dataset['target_annotations_folder'])  # 读取target_annotations_folder文件夹下的全部文件名
    images_data_list = []
    images_data_dict = {}
    for target_annotation in dataset['check_file_name_list']:
        if target_annotation != 'instances_train2017.json':
            continue
        target_annotation_path = os.path.join(
            dataset['target_annotations_folder'], target_annotation)
        print('Loading instances_train2017.json:')
        with open(target_annotation_path, 'r') as f:
            data = json.loads(f.read())
        name_dict = {}
        for one_name in data['categories']:
            name_dict['%s' % one_name['id']] = one_name['name']

        print('Start to count images:')
        total_image_count = 0
        for d in tqdm(data['images']):
            total_image_count += 1
        check_images_count = min(
            dataset['target_annotation_check_count'], total_image_count)
        check_image_id_list = [random.randint(
            0, total_image_count)for i in range(check_images_count)]

        print('Start to load each annotation data file:')
        for n in check_image_id_list:
            d = data['images'][n]
            img_id = d['id']
            img_name = d['file_name']
            img_name_new = img_name
            img_path = os.path.join(
                dataset['temp_images_folder'], img_name_new)
            img = cv2.imread(img_path)
            _, _, channels = img.shape
            width = d['width']
            height = d['height']
            # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
            # 并将初始化后的对象存入total_images_data_list
            one_image = IMAGE(
                img_name, img_name_new, img_path, height, width, channels, [], [])
            images_data_dict.update({img_id: one_image})
        
        for one_annotation in tqdm(data['annotations']):
            if one_annotation['image_id'] in images_data_dict:
                if len(one_annotation['bbox']):
                    ann_image_id = one_annotation['image_id']   # 获取此标签图片id
                    # 获取标签类别
                    cls = name_dict[str(one_annotation['category_id'])]
                    cls = cls.replace(' ', '').lower()
                    if cls not in (dataset['detect_class_list_new']
                                   + dataset['segment_class_list_new']):
                        continue
                    box = [one_annotation['bbox'][0],
                           one_annotation['bbox'][1],
                           one_annotation['bbox'][0] +
                           one_annotation['bbox'][2],
                           one_annotation['bbox'][1] + one_annotation['bbox'][3]]
                    xmin = max(min(int(box[0]), int(box[2]),
                                   int(images_data_dict[ann_image_id].width)), 0.)
                    ymin = max(min(int(box[1]), int(box[3]),
                                   int(images_data_dict[ann_image_id].height)), 0.)
                    xmax = min(max(int(box[2]), int(box[0]), 0.),
                               int(images_data_dict[ann_image_id].width))
                    ymax = min(max(int(box[3]), int(box[1]), 0.),
                               int(images_data_dict[ann_image_id].height))
                    images_data_dict[ann_image_id].true_box_list_updata(
                        TRUE_BOX(cls, xmin, ymin, xmax, ymax))
                if len(one_annotation['segmentation']):
                    ann_image_id = one_annotation['image_id']   # 获取此标签图片id
                    # 获取标签类别
                    cls = name_dict[str(one_annotation['category_id'])]
                    cls = cls.replace(' ', '').lower()
                    if cls not in (dataset['detect_class_list_new']
                                   + dataset['segment_class_list_new']):
                        continue
                    segmentation = np.asarray(
                        one_annotation['segmentation'][0]).reshape((-1, 2)).tolist()
                    images_data_dict[ann_image_id].true_segmentation_list_updata(
                        TRUE_SEGMENTATION(cls, segmentation, iscrowd=one_annotation['iscrowd']))
    for _, n in images_data_dict.items():
        images_data_list.append(n)
    random.shuffle(images_data_list)
    check_images_count = min(
        dataset['target_annotation_check_count'], len(images_data_list))
    check_images_list = images_data_list[0:check_images_count]

    return check_images_list


def cityscapesval(dataset: dict) -> list:
    """[读取cityscapes数据集图片类检测列表]

    Args:
        dataset (dict): [数据集信息字典]

    Returns:
        list: [数据集图片类检测列表]
    """

    return []
