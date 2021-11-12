# -*- coding: utf-8 -*-
import os
from utils.utils import *


def hy_output_folds_list(output_path):
    """返回输出混合数据集的images、labels、ImageSets路径字典"""

    output_path_dict = {}
    output_path_dict['images_path'] = check_output_path(output_path, 'images')
    output_path_dict['labels_path'] = check_output_path(output_path, 'labels')
    output_path_dict['imagesets_path'] = check_output_path(output_path, 'ImageSets')
    output_path_dict['annotations_path'] = check_output_path(output_path, 'Annotations')
    output_path_dict['source_label_path'] = check_output_path(output_path, 'source label')

    return output_path_dict


def func_None(*args):
    """如无对应model的fold函数，需添加函数"""

    print("\nCannot find function, you shoule appen the function.")
    return 0


new_set_fold_path_func_dict = {"ldp": hy_output_folds_list, "hy": hy_output_folds_list}


def pickup_new_set_output_function(model, *args):
    """根据输入类别挑选更换图片文件名"""

    return new_set_fold_path_func_dict.get(model, func_None)(*args)  # 返回对应类别更名函数
