'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2021-11-12 18:05:15
'''
import os
import json
import xml.etree.ElementTree as ET

from annotation.annotation_temp import TEMP_LOAD


def annotation_creat_root(dataset: dict, color_list: list) -> None:
    """[创建xml根节点]

    Args:
        dataset (dict): [数据集信息字典]
        color_list (list): [色彩列表]

    Returns:
        [type]: [xml根节点]
    """    
    class_id = 0
    annotations = ET.Element('annotations')
    version = ET.SubElement(annotations, 'version')
    version.text = '1.1'
    meta = ET.SubElement(annotations, 'meta')
    task = ET.SubElement(meta, 'task')
    # ET.SubElement(task, 'id')
    # ET.SubElement(task, 'name')
    # ET.SubElement(task, 'size')
    # mode = ET.SubElement(task, 'mode')
    # mode.text = 'annotation'
    # overlap = ET.SubElement(task, 'overlap')
    # ET.SubElement(task, 'bugtracker')
    # ET.SubElement(task, 'created')
    # ET.SubElement(task, 'updated')
    # subset = ET.SubElement(task, 'subset')
    # subset.text = 'default'
    # start_frame = ET.SubElement(task, 'start_frame')
    # start_frame.text='0'
    # ET.SubElement(task, 'stop_frame')
    # ET.SubElement(task, 'frame_filter')
    # segments = ET.SubElement(task, 'segments')
    # segment = ET.SubElement(segments, 'segment')
    # ET.SubElement(segments, 'id')
    # start = ET.SubElement(segments, 'start')
    # start.text='0'
    # ET.SubElement(segments, 'stop')
    # ET.SubElement(segments, 'url')
    # owner = ET.SubElement(task, 'owner')
    # ET.SubElement(owner, 'username')
    # ET.SubElement(owner, 'email')
    # ET.SubElement(task, 'assignee')
    labels = ET.SubElement(task, 'labels')

    class_dict_list_output_path = os.path.join(
        dataset['target_annotations_folder'], 'class_dict_list.txt')
    with open(class_dict_list_output_path, 'w') as f:
        for n, c in zip(dataset['class_list_new'], color_list):
            label = ET.SubElement(labels, 'label')
            name = ET.SubElement(label, 'name')
            name.text = n
            color = ET.SubElement(label, 'color')
            color.text = c
            attributes = ET.SubElement(label, 'attributes')
            attribute = ET.SubElement(attributes, 'attribute')
            name = ET.SubElement(attribute, 'name')
            name.text = '1'
            mutable = ET.SubElement(attribute, 'mutable')
            mutable.text = 'False'
            input_type = ET.SubElement(attribute, 'input_type')
            input_type.text = 'text'
            default_value = ET.SubElement(attribute, 'default_value')
            default_value.text = None
            values = ET.SubElement(attribute, 'values')
            values.text = None
            # 输出标签色彩txt
            s = '  {\n    "name": "'+n+'",\n    "color": "' + \
                str(c)+'",\n    "attributes": []\n  },\n'
            f.write(s)
            class_id += 1

        # ET.SubElement(task, 'dumped')
    return annotations


def annotation_get_temp(dataset: dict, temp_annotation_path: str) -> None:
    """[获取temp标签信息]

    Args:
        dataset (dict): [数据集信息字典]
        temp_annotation_path (str): [暂存标签路径]
    """
    image = TEMP_LOAD(dataset, temp_annotation_path)
    if image == None:
        return
    image_xml = ET.Element('image', {
        'id': '', 'name': image.image_name_new, 'width': str(image.width), 'height': str(image.height)})
    for n in image.true_segmentation_list:
        point_list = []
        for x in n.segmentation:
            point_list.append(str(x[0])+','+str(x[1]))
        if 2 == len(point_list):
            continue
        polygon = ET.SubElement(image_xml, 'polygon', {
                                'label': n.clss, 'occluded': '0', 'source': 'manual', 'points': ';'.join(point_list)})
        attribute = ET.SubElement(polygon, 'attribute', {'name': '1'})

    return image_xml


def annotation_output(dataset: dict, temp_annotation_path: str, process_output) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        temp_annotation_path (str): [暂存标签路径]
    """

    image = TEMP_LOAD(dataset, temp_annotation_path)
    if image == None:
        return
    annotation_output_path = os.path.join(
        dataset['target_annotations_folder'], image.file_name + '.' + dataset['target_annotation_form'])
    annotation = {'imgHeight': image.height,
                  'imgWidth': image.width,
                  'objects': []
                  }
    segmentation = {}
    for true_segmentation in image.true_segmentation_list:
        segmentation = {'label': true_segmentation.clss,
                        'polygon': true_segmentation.segmentation
                        }
        annotation['objects'].append(segmentation)
    json.dump(annotation, open(annotation_output_path, 'w'))

    return
