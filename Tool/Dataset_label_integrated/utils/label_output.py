# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from utils.utils import *
from utils.output_path_function import *
from utils.input_path_function import *
import shutil


def label(fix_total_label_path_dict, new_class_names_list, src_set_style, output_path_dict):
    """向新数据集路径写入修改后的标签"""

    image_type = check_image_type(src_set_style)    # 获取源数据集图片类型
    ouput_dataset_labels_path = output_path_dict['labels_path']

    total_mix_images_name_list = []     # 创建混合数据集包含的源数据集图片路径列表
    for key, value in tqdm(fix_total_label_path_dict.items()):  # 遍历labels和bbox字典
        # 将对应key的label名称和bbox存入修改后的label中
        label_output_path = os.path.join(ouput_dataset_labels_path, key)
        with open(label_output_path, 'w') as out_label:
            for one_fix_bbox in value:
                out_label.write(
                    " ".join([str(a) for a in one_fix_bbox]) + '\n')
        total_mix_images_name_list.append(label_output_path.split('\\')[-1].replace('.txt', ''))

    return total_mix_images_name_list