'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-05 21:50:49
LastEditors: Leidi
LastEditTime: 2021-12-22 11:02:39
'''
import os
import json
import time
import random
from tqdm import tqdm
import multiprocessing
import xml.etree.ElementTree as ET

from base.image_base import *
import annotation.dataset_output_function as F
from utils.utils import err_call_back, RGB_to_Hex


def cityscapes(dataset: dict) -> None:
    """[输出temp dataset annotation为CITYSCAPES]

     Args:
         dataset (dict): [temp dataset]
    """

    print('Start output target annotations:')
    pool = multiprocessing.Pool(dataset['workers'])
    for temp_annotation_path in tqdm(dataset['temp_annotation_path_list']):
        pool.apply_async(func=F.__dict__[dataset['target_dataset_style']].annotation_output,
                         args=(dataset, temp_annotation_path,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    return


def cityscapes_val(dataset: dict) -> None:
    """[输出temp dataset annotation为CITYSCAPESVAL]

     Args:
         dataset (dict): [数据集信息字典]
    """

    print('Start output target annotations:')
    pool = multiprocessing.Pool(dataset['workers'])
    for temp_annotation_path in tqdm(dataset['temp_annotation_path_list']):
        pool.apply_async(func=F.__dict__[dataset['target_dataset_style']].annotation_output,
                         args=(dataset, temp_annotation_path,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    return


def cvat_image_1_1(dataset: dict) -> None:
    """[输出temp dataset annotation为cvat_image_1_1]

     Args:
         dataset (dict): [数据集信息字典]
    """

    # 获取不重复随机颜色编码
    print('Start output target annotations:')
    encode = []
    for n in range(len(dataset['class_list_new'])):
        encode.append(random.randint(0, 255))
    # 转换不重复随机颜色编码为16进制颜色
    color_list = []
    for n in encode:
        color_list.append(RGB_to_Hex(str(n)+','+str(n)+','+str(n)))

    # 生成空基本信息xml文件
    annotations = F.__dict__[dataset['target_dataset_style']
                             ].annotation_creat_root(dataset, color_list)
    # 获取全部图片标签信息列表
    total_image_xml = []
    pool = multiprocessing.Pool(dataset['workers'])
    for temp_annotation_path in tqdm(dataset['temp_annotation_path_list']):
        total_image_xml.append(pool.apply_async(func=F.__dict__[dataset['target_dataset_style']].annotation_get_temp,
                                                args=(
                                                    dataset, temp_annotation_path,),
                                                error_callback=err_call_back))
    pool.close()
    pool.join()

    # 将image标签信息添加至annotations中
    for n, image in enumerate(total_image_xml):
        annotation = image.get()
        annotation.attrib['id'] = str(n)
        annotations.append(annotation)

    tree = ET.ElementTree(annotations)

    annotation_output_path = os.path.join(
        dataset['target_annotations_folder'], 'annotatons.' + dataset['target_annotation_form'])
    tree.write(annotation_output_path, encoding='utf-8', xml_declaration=True)

    return


def coco2017(dataset) -> None:
    """[输出temp dataset annotation]

    Args:
        dataset (Dataset): [temp dataset]
    """

    print('Start output target annotations:')
    for dataset_temp_annotation_path_list in tqdm(dataset['temp_divide_file_list'][1:-1]):
        # 声明coco字典及基础信息
        coco = {'info': {'description': 'COCO 2017 Dataset',
                         'url': 'http://cocodataset.org',
                         'version': '1.0',
                         'year': 2017,
                         'contributor': 'leidi',
                         'date_created': time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
                         },
                'licenses': [
            {
                'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
                'id': 1,
                'name': 'Attribution-NonCommercial-ShareAlike License'
            },
            {
                'url': 'http://creativecommons.org/licenses/by-nc/2.0/',
                'id': 2,
                'name': 'Attribution-NonCommercial License'
            },
            {
                'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/',
                'id': 3,
                'name': 'Attribution-NonCommercial-NoDerivs License'
            },
            {
                'url': 'http://creativecommons.org/licenses/by/2.0/',
                'id': 4,
                'name': 'Attribution License'
            },
            {
                'url': 'http://creativecommons.org/licenses/by-sa/2.0/',
                'id': 5,
                'name': 'Attribution-ShareAlike License'
            },
            {
                'url': 'http://creativecommons.org/licenses/by-nd/2.0/',
                'id': 6,
                'name': 'Attribution-NoDerivs License'
            },
            {
                'url': 'http://flickr.com/commons/usage/',
                'id': 7,
                'name': 'No known copyright restrictions'
            },
            {
                'url': 'http://www.usa.gov/copyright.shtml',
                'id': 8,
                'name': 'United States Government Work'
            }
        ],
            'images': [],
            'annotations': [],
            'categories': []
        }

        # 将class_list_new转换为coco格式字典
        for n, cls in enumerate(dataset['class_list_new']):
            category_item = {'supercategory': 'none',
                             'id': n,
                             'name': cls}
            coco['categories'].append(category_item)

        annotation_output_path = os.path.join(
            dataset['target_annotations_folder'], os.path.splitext(
                dataset_temp_annotation_path_list.split(os.sep)[-1])[0]
            + str(2017) + '.' + dataset['target_annotation_form'])
        annotation_path_list = []
        with open(dataset_temp_annotation_path_list, 'r') as f:
            for n in f.readlines():
                annotation_path_list.append(n.replace('\n', '')
                                            .replace(dataset['source_images_folder'], dataset['temp_annotations_folder'])
                                            .replace(dataset['target_image_form'], dataset['temp_annotation_form']))

        # 读取标签图片基础信息
        print('Start load image information:')
        image_information_list = []
        pool = multiprocessing.Pool(dataset['workers'])
        for n, temp_annotation_path in tqdm(enumerate(annotation_path_list)):
            image_information_list.append(pool.apply_async(func=F.__dict__[dataset['target_dataset_style']].get_image_information,
                                                           args=(
                                                               dataset, coco, n, temp_annotation_path,),
                                                           error_callback=err_call_back))
        pool.close()
        pool.join()

        for n in tqdm(image_information_list):
            coco['images'].append(n.get())
        del image_information_list

        # 读取标签标注基础信息
        print('Start load annotation:')
        annotations_list = []
        pool = multiprocessing.Pool(dataset['workers'])
        process_annotation_count = multiprocessing.Manager().dict(
            {'annotation_count': 0})
        for n, temp_annotation_path in tqdm(enumerate(annotation_path_list)):
            annotations_list.append(pool.apply_async(func=F.__dict__[dataset['target_dataset_style']].get_annotation,
                                                     args=(dataset, n,
                                                           temp_annotation_path, process_annotation_count,),
                                                     error_callback=err_call_back))
        pool.close()
        pool.join()

        for n in tqdm(annotations_list):
            for m in n.get():
                coco['annotations'].append(m)
        del annotations_list

        json.dump(coco, open(annotation_output_path, 'w'))

    return
