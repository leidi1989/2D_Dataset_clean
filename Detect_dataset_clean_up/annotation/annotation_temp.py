'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-09 00:59:33
LastEditors: Leidi
LastEditTime: 2021-10-22 16:39:36
'''
import os
import cv2
import codecs
import xml.etree.ElementTree as ET

from base.image_base import *


def TEMP_LOAD(dataset: dict, temp_annotation_path: str) -> IMAGE:
    """[读取暂存annotation]

    Args:
        dataset (dict): [数据集信息字典]
        temp_annotation_path (str): [暂存标注文件路径]

    Returns:
        IMAGE: [IMAGE类实例]
    """

    tree = ET.parse(temp_annotation_path)
    root = tree.getroot()
    image_name = str(root.find('filename').text)
    image_path = os.path.join(dataset['source_images_folder'], image_name)
    image_size = cv2.imread(image_path).shape
    height = image_size[0]
    width = image_size[1]
    channels = image_size[2]
    truebox_dict_list = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = str(obj.find('name').text)
        cls = cls.replace(' ', '').lower()
        if cls not in dataset['class_list_new']:
            continue
        if int(difficult) == 1:
            continue
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        xmin = max(min(float(b[0]), float(b[1]), float(width)), 0.)
        ymin = max(min(float(b[2]), float(b[3]), float(height)), 0.)
        xmax = min(max(float(b[1]), float(b[0]), 0.), float(width))
        ymax = min(max(float(b[3]), float(b[2]), 0.), float(height))
        truebox_dict_list.append(TRUE_BOX(
            cls, xmin, ymin, xmax, ymax, 'rectangle', difficult))  # 将单个真实框加入单张图片真实框列表
    one_image = IMAGE(image_name, image_name, image_path, int(
        height), int(width), int(channels), truebox_dict_list)

    return one_image


def TEMP_OUTPUT(annotation_output_path: str, image: IMAGE) -> bool:
    """[输出temp dataset annotation]

    Args:
        annotation_output_path (str): [temp dataset annotation输出路径]
        image (IMAGE): [IMAGE实例]
        class_names_list_new ([type], optional): [类别修改后的类别列表]. Defaults to None.
    """

    if image == None:
        return False

    with codecs.open(annotation_output_path, "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
        xml.write('\t<filename>' + image.image_name_new + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>The VOC2007 Database</database>\n')
        xml.write('\t\t<annotation>PASCAL VOC2007</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>WH</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(int(image.width)) + '</width>\n')
        xml.write('\t\t<height>' +
                  str(int(image.height)) + '</height>\n')
        xml.write('\t\t<depth>' +
                  str(int(image.channels)) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        if len(image.true_box_list):
            for box in image.true_box_list:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + box.clss + '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' +
                          str(int(box.xmin)) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' +
                          str(int(box.ymin)) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' +
                          str(int(box.xmax)) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' +
                          str(int(box.ymax)) + '</ymax>\n')
                # TODO: 距离和遮挡
                # xml.write('\t\t\t<distance>' +
                #           str(int(box.distance)) + '</distance>\n')
                # xml.write('\t\t\t<occlusion>' +
                #           str(float(box.occlusion)) + '</occlusion>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
        xml.write('</annotation>')
        xml.close()

    return True
