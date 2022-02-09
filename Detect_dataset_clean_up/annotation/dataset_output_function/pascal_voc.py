'''
Description:
Version:
Author: Leidi
Date: 2021-10-19 15:55:16
LastEditors: Leidi
LastEditTime: 2021-10-22 16:27:13
'''
import os
import codecs

from annotation.annotation_temp import TEMP_LOAD


def annotation_output(dataset: dict, temp_annotation_path: str) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        temp_annotation_path (str): [暂存标签路径]
    """

    image = TEMP_LOAD(dataset, temp_annotation_path)
    annotation_output_path = os.path.join(
        dataset['target_annotations_folder'], image.file_name + '.' + dataset['target_annotation_form'])
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
    
    return
            
