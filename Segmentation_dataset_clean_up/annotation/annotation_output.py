'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-05 21:50:49
LastEditors: Leidi
LastEditTime: 2021-12-20 20:05:47
'''
import os
import json
import random
from tqdm import tqdm
import multiprocessing
import xml.etree.ElementTree as ET

from base.image_base import *
import annotation.dataset_output_function as F
from utils.utils import err_call_back, RGB_to_Hex


# COCO转换所需常量
COCO_category_item_id = -1
COCO_image_id = 0
COCO_annotation_id = 0


def coco2017(dataset) -> None:
    """[输出temp dataset annotation为COCO2017]

     Args:
         dataset (Dataset): [temp dataset]
    """

    def addCatItem(name):
        global COCO_category_item_id
        category_item = dict()
        category_item['supercategory'] = 'none'
        COCO_category_item_id += 1
        category_item['id'] = COCO_category_item_id
        category_item['name'] = name
        coco['categories'].append(category_item)
        category_set[name] = COCO_category_item_id
        return COCO_category_item_id

    def addImgItem(file_name, size):
        global COCO_image_id
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        if size['width'] is None:
            raise Exception('Could not find width tag in xml file.')
        if size['height'] is None:
            raise Exception('Could not find height tag in xml file.')
        COCO_image_id += 1
        image_item = dict()
        image_item['id'] = COCO_image_id
        image_item['file_name'] = file_name
        image_item['width'] = size['width']
        image_item['height'] = size['height']
        coco['images'].append(image_item)
        image_set.add(file_name)
        return COCO_image_id

    def addAnnoItem(object_name, image_id, category_id, bbox):
        global COCO_annotation_id
        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg = []
        # bbox[] is x,y,w,h
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])
        annotation_item['segmentation'].append(seg)
        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        COCO_annotation_id += 1
        annotation_item['id'] = COCO_annotation_id
        coco['annotations'].append(annotation_item)

    print('Start output temp dataset annotations to COCO2017 annotation:')
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    category_set = dict()
    for cls in dataset['class_list_new']:
        addCatItem(cls)
    image_set = set()

    divide_list = ['train', 'test']
    print('\nTemp dataset transform to COCO:')
    for n in divide_list:
        image_sets_file = os.path.join(
            dataset['temp_informations_folder'], 'Main', n + '.txt')
        json_save_path = os.path.join(
            dataset['target_annotations_folder'], n + '.json')

        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
            f.close()
        print('Transform to {}.json:'.format(n))
        for _id in tqdm(ids):
            xml_file = os.path.join(
                dataset['temp_annotations_folder'], _id + '.xml')

            bndbox = dict()
            size = dict()
            current_image_id = None
            current_category_id = None
            file_name = None
            size['width'] = None
            size['height'] = None
            size['depth'] = None

            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.tag != 'annotation':
                raise Exception(
                    'pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

            # elem is <folder>, <filename>, <size>, <object>
            for elem in root:
                current_parent = elem.tag
                current_sub = None
                object_name = None

                if elem.tag == 'folder':
                    continue

                if elem.tag == 'filename':
                    file_name = elem.text
                    if file_name in category_set:
                        raise Exception('file_name duplicated')

                # add img item only after parse <size> tag
                elif current_image_id is None and file_name is not None and size['width'] is not None:
                    if file_name not in image_set:
                        current_image_id = addImgItem(file_name, size)
                        # print('add image with {} and {}'.format(file_name, size))
                    else:
                        raise Exception(
                            'duplicated image: {}'.format(file_name))
                        # subelem is <width>, <height>, <depth>, <name>, <bndbox>
                for subelem in elem:
                    bndbox['xmin'] = None
                    bndbox['xmax'] = None
                    bndbox['ymin'] = None
                    bndbox['ymax'] = None

                    current_sub = subelem.tag
                    if current_parent == 'object' and subelem.tag == 'name':
                        object_name = subelem.text
                        if object_name not in category_set:
                            current_category_id = addCatItem(object_name)
                        else:
                            current_category_id = category_set[object_name]

                    elif current_parent == 'size':
                        if size[subelem.tag] is not None:
                            raise Exception(
                                'xml structure broken at size tag.')
                        size[subelem.tag] = int(subelem.text)

                    # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                    for option in subelem:
                        if current_sub == 'bndbox':
                            if bndbox[option.tag] is not None:
                                raise Exception(
                                    'xml structure corrupted at bndbox tag.')
                            bndbox[option.tag] = int(option.text)

                    # only after parse the <object> tag
                    if bndbox['xmin'] is not None:
                        if object_name is None:
                            raise Exception(
                                'xml structure broken at bndbox tag')
                        if current_image_id is None:
                            raise Exception(
                                'xml structure broken at bndbox tag')
                        if current_category_id is None:
                            raise Exception(
                                'xml structure broken at bndbox tag')
                        bbox = []
                        # x
                        bbox.append(bndbox['xmin'])
                        # y
                        bbox.append(bndbox['ymin'])
                        # w
                        bbox.append(bndbox['xmax'] - bndbox['xmin'])
                        # h
                        bbox.append(bndbox['ymax'] - bndbox['ymin'])
                        # print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                        #                                                bbox))
                        addAnnoItem(object_name, current_image_id,
                                    current_category_id, bbox)
        json.dump(coco, open(json_save_path, 'w'))

    return


def cityscapes(dataset: dict) -> None:
    """[输出temp dataset annotation为CITYSCAPES]

     Args:
         dataset (dict): [temp dataset]
    """

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




def COCO_2017_OUTPUT(dataset) -> None:
    """[输出temp dataset annotation]

    Args:
        dataset (Dataset): [temp dataset]
    """
    # coco转换所需常量
    category_item_id = -1
    image_id = 0
    annotation_id = 0

    data
    
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    category_set = dict()
    for cls in dataset['class_list_new']:
        addCatItem(cls)
    image_set = set()

    divide_list = ['train', 'test', 'val']
    print('\nTemp dataset transform to COCO:')
    for n in divide_list:
        image_sets_file = os.path.join(
            dataset['temp_informations_folder'], 'Main', n + '.txt')
        json_save_path = os.path.join(
            dataset['target_annotations_folder'], n + '.json')

        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
            f.close()
        print('Transform to {}.json:'.format(n))
        for _id in tqdm(ids):
            xml_file = os.path.join(
                dataset['temp_annotations_folder'], _id + '.xml')

            bndbox = dict()
            size = dict()
            current_image_id = None
            current_category_id = None
            file_name = None
            size['width'] = None
            size['height'] = None
            size['depth'] = None

            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.tag != 'annotation':
                raise Exception(
                    'pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

            # elem is <folder>, <filename>, <size>, <object>
            for elem in root:
                current_parent = elem.tag
                current_sub = None
                object_name = None

                if elem.tag == 'folder':
                    continue

                if elem.tag == 'filename':
                    file_name = elem.text
                    if file_name in category_set:
                        raise Exception('file_name duplicated')

                # add img item only after parse <size> tag
                elif current_image_id is None and file_name is not None and size['width'] is not None:
                    if file_name not in image_set:
                        current_image_id = addImgItem(file_name, size)
                        # print('add image with {} and {}'.format(file_name, size))
                    else:
                        raise Exception(
                            'duplicated image: {}'.format(file_name))
                        # subelem is <width>, <height>, <depth>, <name>, <bndbox>
                for subelem in elem:
                    bndbox['xmin'] = None
                    bndbox['xmax'] = None
                    bndbox['ymin'] = None
                    bndbox['ymax'] = None

                    current_sub = subelem.tag
                    if current_parent == 'object' and subelem.tag == 'name':
                        object_name = subelem.text
                        if object_name not in category_set:
                            current_category_id = addCatItem(object_name)
                        else:
                            current_category_id = category_set[object_name]

                    elif current_parent == 'size':
                        if size[subelem.tag] is not None:
                            raise Exception(
                                'xml structure broken at size tag.')
                        size[subelem.tag] = int(subelem.text)

                    # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                    for option in subelem:
                        if current_sub == 'bndbox':
                            if bndbox[option.tag] is not None:
                                raise Exception(
                                    'xml structure corrupted at bndbox tag.')
                            bndbox[option.tag] = int(option.text)

                    # only after parse the <object> tag
                    if bndbox['xmin'] is not None:
                        if object_name is None:
                            raise Exception(
                                'xml structure broken at bndbox tag')
                        if current_image_id is None:
                            raise Exception(
                                'xml structure broken at bndbox tag')
                        if current_category_id is None:
                            raise Exception(
                                'xml structure broken at bndbox tag')
                        bbox = []
                        # x
                        # bbox.append(bndbox['xmin'])
                        # # y
                        # bbox.append(bndbox['ymin'])
                        # # w
                        # bbox.append(bndbox['xmax'] - bndbox['xmin'])
                        # # h
                        # bbox.append(bndbox['ymax'] - bndbox['ymin'])
                        # print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                        #                                                bbox))
                        segmentation = []
                        addAnnoItem(object_name, current_image_id,
                                    current_category_id, bbox, )
        json.dump(coco, open(json_save_path, 'w'))

    return


# category_item_id = -1
# image_id = 0
# annotation_id = 0


# def COCO_2017_OUTPUT(dataset) -> None:
#     """[输出temp dataset annotation]

#     Args:
#         dataset (Dataset): [temp dataset]
#     """

#     def addCatItem(name):
#         global category_item_id
#         category_item = dict()
#         category_item['supercategory'] = 'none'
#         category_item_id += 1
#         category_item['id'] = category_item_id
#         category_item['name'] = name
#         coco['categories'].append(category_item)
#         category_set[name] = category_item_id
#         return category_item_id

#     def addImgItem(file_name, size):
#         global image_id
#         if file_name is None:
#             raise Exception('Could not find filename tag in xml file.')
#         if size['width'] is None:
#             raise Exception('Could not find width tag in xml file.')
#         if size['height'] is None:
#             raise Exception('Could not find height tag in xml file.')
#         image_id += 1
#         image_item = dict()
#         image_item['id'] = image_id
#         image_item['file_name'] = file_name
#         image_item['width'] = size['width']
#         image_item['height'] = size['height']
#         coco['images'].append(image_item)
#         image_set.add(file_name)
#         return image_id

#     def addAnnoItem(object_name, image_id, category_id, bbox, segmentation):
#         global annotation_id
#         annotation_item = dict()
#         annotation_item['segmentation'] = []
#         seg = []
#         # bbox[] is x,y,w,h
#         # left_top
#         seg.append(bbox[0])
#         seg.append(bbox[1])
#         # left_bottom
#         seg.append(bbox[0])
#         seg.append(bbox[1] + bbox[3])
#         # right_bottom
#         seg.append(bbox[0] + bbox[2])
#         seg.append(bbox[1] + bbox[3])
#         # right_top
#         seg.append(bbox[0] + bbox[2])
#         seg.append(bbox[1])
#         annotation_item['segmentation'].append(seg)
#         annotation_item['area'] = bbox[2] * bbox[3]
#         annotation_item['iscrowd'] = 0
#         annotation_item['ignore'] = 0
#         annotation_item['image_id'] = image_id
#         annotation_item['bbox'] = bbox
#         annotation_item['category_id'] = category_id
#         annotation_id += 1
#         annotation_item['id'] = annotation_id
#         coco['annotations'].append(annotation_item)

#     coco = dict()
#     coco['images'] = []
#     coco['type'] = 'instances'
#     coco['annotations'] = []
#     coco['categories'] = []
#     category_set = dict()
#     for cls in dataset['class_list_new']:
#         addCatItem(cls)
#     image_set = set()

#     divide_list = ['train', 'test', 'val']
#     print('\nTemp dataset transform to COCO:')
#     for n in divide_list:
#         image_sets_file = os.path.join(
#             dataset['temp_informations_folder'], 'Main', n + '.txt')
#         json_save_path = os.path.join(
#             dataset['target_annotations_folder'], n + '.json')

#         ids = []
#         with open(image_sets_file) as f:
#             for line in f:
#                 ids.append(line.rstrip())
#             f.close()
#         print('Transform to {}.json:'.format(n))
#         for _id in tqdm(ids):
#             xml_file = os.path.join(
#                 dataset['temp_annotations_folder'], _id + '.xml')

#             bndbox = dict()
#             size = dict()
#             current_image_id = None
#             current_category_id = None
#             file_name = None
#             size['width'] = None
#             size['height'] = None
#             size['depth'] = None

#             tree = ET.parse(xml_file)
#             root = tree.getroot()
#             if root.tag != 'annotation':
#                 raise Exception(
#                     'pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

#             # elem is <folder>, <filename>, <size>, <object>
#             for elem in root:
#                 current_parent = elem.tag
#                 current_sub = None
#                 object_name = None

#                 if elem.tag == 'folder':
#                     continue

#                 if elem.tag == 'filename':
#                     file_name = elem.text
#                     if file_name in category_set:
#                         raise Exception('file_name duplicated')

#                 # add img item only after parse <size> tag
#                 elif current_image_id is None and file_name is not None and size['width'] is not None:
#                     if file_name not in image_set:
#                         current_image_id = addImgItem(file_name, size)
#                         # print('add image with {} and {}'.format(file_name, size))
#                     else:
#                         raise Exception(
#                             'duplicated image: {}'.format(file_name))
#                         # subelem is <width>, <height>, <depth>, <name>, <bndbox>
#                 for subelem in elem:
#                     bndbox['xmin'] = None
#                     bndbox['xmax'] = None
#                     bndbox['ymin'] = None
#                     bndbox['ymax'] = None

#                     current_sub = subelem.tag
#                     if current_parent == 'object' and subelem.tag == 'name':
#                         object_name = subelem.text
#                         if object_name not in category_set:
#                             current_category_id = addCatItem(object_name)
#                         else:
#                             current_category_id = category_set[object_name]

#                     elif current_parent == 'size':
#                         if size[subelem.tag] is not None:
#                             raise Exception(
#                                 'xml structure broken at size tag.')
#                         size[subelem.tag] = int(subelem.text)

#                     # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
#                     for option in subelem:
#                         if current_sub == 'bndbox':
#                             if bndbox[option.tag] is not None:
#                                 raise Exception(
#                                     'xml structure corrupted at bndbox tag.')
#                             bndbox[option.tag] = int(option.text)

#                     # only after parse the <object> tag
#                     if bndbox['xmin'] is not None:
#                         if object_name is None:
#                             raise Exception(
#                                 'xml structure broken at bndbox tag')
#                         if current_image_id is None:
#                             raise Exception(
#                                 'xml structure broken at bndbox tag')
#                         if current_category_id is None:
#                             raise Exception(
#                                 'xml structure broken at bndbox tag')
#                         bbox = []
#                         # x
#                         # bbox.append(bndbox['xmin'])
#                         # # y
#                         # bbox.append(bndbox['ymin'])
#                         # # w
#                         # bbox.append(bndbox['xmax'] - bndbox['xmin'])
#                         # # h
#                         # bbox.append(bndbox['ymax'] - bndbox['ymin'])
#                         # print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
#                         #                                                bbox))
#                         segmentation = []
#                         addAnnoItem(object_name, current_image_id,
#                                     current_category_id, bbox, )
#         json.dump(coco, open(json_save_path, 'w'))

#     return
