# -*- coding: utf-8 -*-
import os
from utils.utils import *


def hy_input_folds_list(input_path):
    """返回输出混合数据集的images、labels、ImageSets路径字典"""

    input_path_dict = {}
    input_path_dict['images_path'] = check_input_path(os.path.join(input_path, 'images'))
    input_path_dict['labels_path'] = check_input_path(os.path.join(input_path, 'labels'))
    input_path_dict['imagesets_path'] = check_input_path(os.path.join(input_path, 'ImageSets'))
    input_path_dict['annotations_path'] = check_input_path(os.path.join(input_path, 'Annotations'))
    input_path_dict['source_label_path'] = check_input_path(os.path.join(input_path, 'source label'))

    return input_path_dict


def func_None(*args):
    """如无对应model的fold函数，需添加函数"""

    print("\nCannot find function, you shoule appen the function.")
    return 0


src_set_fold_path_func_dict = {"ldp": hy_input_folds_list, "hy": hy_input_folds_list, "kitti": hy_input_folds_list}


def pickup_src_set_output_function(model, *args):
    """根据输入类别挑选更换图片文件名"""

    return src_set_fold_path_func_dict.get(model, func_None)(*args)  # 返回对应类别更名函数
