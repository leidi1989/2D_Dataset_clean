'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-19 20:56:09
LastEditors: Leidi
LastEditTime: 2021-10-22 17:35:12
'''
import os

from utils.convertion_function import to_yolo
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
    one_image_bbox = []                                     # 声明每张图片bbox列表
    for true_box in image.true_box_list:                        # 遍历单张图片全部bbox
        true_box_class = str(true_box.clss).replace(
            ' ', '').lower()    # 获取bbox类别
        if true_box_class in set(dataset['class_list_new']):
            cls_id = dataset['class_list_new'].index(true_box_class)
            b = (true_box.xmin, true_box.xmax, true_box.ymin,
                 true_box.ymax,)                                # 获取源标签bbox的xxyy
            bb = to_yolo((image.width, image.height), b)       # 转换bbox至yolo类型
            one_image_bbox.append([cls_id, bb])
        else:
            print('\nErro! Class not in classes.names image: %s!' %
                  image.image_name)

    with open(annotation_output_path, 'w') as f:   # 创建图片对应txt格式的label文件
        for one_bbox in one_image_bbox:
            f.write(str(one_bbox[0]) + " " +
                    " ".join([str(a) for a in one_bbox[1]]) + '\n')
        f.close()

    return
