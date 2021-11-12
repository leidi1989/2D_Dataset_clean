# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from utils.utils import *
from utils.output_path_function import *
from utils.input_path_function import *
import shutil


def imagesets_flod(output_path_dict, total_mix_images_name_list, src_set_style, new_class_names_list):
    """为混合数据集imageset内添加total.txt、total_image_name.txt文件"""

    image_type = check_image_type(src_set_style)    # 获取源数据集图片类型
    
    # 输出图片路径
    print('\nStart to creat mix set total.txt:')
    with open(os.path.join(output_path_dict['imagesets_path'], 'total.txt'), 'w') as total_txt:
        for one_image_path in tqdm(total_mix_images_name_list):
            one_image_path = os.path.join(output_path_dict['images_path'], one_image_path + '.' + image_type)
            total_txt.write(one_image_path + '\n')

    # 输出不含后缀的图片名称total_image_name.txt
    print('\nStart to creat mix set total_image_name.txt:')
    with open(os.path.join(output_path_dict['imagesets_path'], 'total_image_name.txt'), 'w') as total_image_name_txt:
        for one_image_name in tqdm(total_mix_images_name_list):
            total_image_name_txt.write(one_image_name + '\n')

    # 输出混合数据集classes.names
    print('\nStart to creat mix set classes.names:')
    with open(os.path.join(output_path_dict['imagesets_path'], 'classes.names'), 'w') as classes_names:
        for one_class in tqdm(new_class_names_list):
            classes_names.write(one_class + '\n')
