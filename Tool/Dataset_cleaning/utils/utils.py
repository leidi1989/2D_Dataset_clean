'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:03
LastEditors: Leidi
LastEditTime: 2021-12-21 16:20:07
'''
# -*- coding: utf-8 -*-
import os
from tqdm import tqdm


def check_input_path(path):
    """[检查输入路径是否存在]

    Parameters
    ----------
    path : [str]
        [输入路径]

    Returns
    -------
    path : [str]
        [输入路径]
    """

    if os.path.exists(path):
        return path
    else:
        # assert path was found
        assert os.path.exists(path), 'Input path Not Found: %s' % path


def check_output_path(path, attach=''):
    """[检查输出路径是否存在]

    Parameters
    ----------
    path : [str]
        [输出路径]
    attach : str, optional
        [输出路径后缀], by default ''

    Returns
    -------
    os.path.join(path, attach) : [str]
        [添加后缀的输出路径]
    """

    if os.path.exists(os.path.join(path, attach)):
        return os.path.join(path, attach)
    else:
        print('Creat output path: %s\n' % str(os.path.join(path, attach)))
        os.makedirs(os.path.join(path, attach))
        return os.path.join(path, attach)


def check_image_type(input_label_style):
    """[确定输入图片格式]

    Parameters
    ----------
    input_label_style : [str]
        [源数据集类别]

    Returns
    -------
    file_type : [str]
        [源数据集图片类别]
    """

    jpg_list = ['ldp', 'hy', 'myxb', 'voc', 
                'coco2017', 'pascal', 'sjt', 'yolov5', 
                'nuscenes', 'compcars', 'yolov5_detect',
                'ccpd', 'yolo', 'licenseplate']
    png_list = ['hy_highway', 'kitti', 'cctsdb']

    if not (input_label_style in jpg_list,
            input_label_style in png_list):      # 判断输入的类型在不在已辨认列表中
        print("\n无对应输出图像格式，请添加输出图像格式！\n")

    if input_label_style in jpg_list:
        file_type = 'jpg'
    if input_label_style in png_list:
        file_type = 'png'

    return file_type


def check_src_lab_type(input_label_style):
    """[确定输入label格式]

    Parameters
    ----------
    input_label_style : [str]
        [源数据集类别]

    Returns
    -------
    file_type : [str]
        [标签文件类别]
    """

    json_list = ['hy', 'myxb', 'coco2017', 'sjt', 'nuscenes']
    xml_list = ['ldp', 'hy_highway', 'pascal']
    txt_list = ['kitti', 'yolo', 'yolov5', 'cctsdb', 'yolov5_detect', 'licenseplate']
    jpg_list = ['ccpd']
    file_type = ''
    if not (input_label_style in json_list,
            input_label_style in xml_list,
            input_label_style in txt_list,
            input_label_style in jpg_list):      # 判断输入的类型在不在已辨认列表中
        print("\n无对应输出label格式，请添加输出label格式")

    if input_label_style in json_list:
        file_type = 'json'
    if input_label_style in xml_list:
        file_type = 'xml'
    if input_label_style in txt_list:
        file_type = 'txt'
    if input_label_style in jpg_list:
        file_type = 'jpg'

    return file_type


def check_pref(pref, segment):
    """[检查是否添加前缀]

    Parameters
    ----------
    pref : [str]
        [前缀字符串]

    Returns
    -------
    name_pre : [str]
        [前缀字符串]
    """

    if pref == '':
        name_pre = pref
    else:
        name_pre = pref + segment

        return name_pre


def out_func_None(*args):
    """如无对应输出标签函数，提示用户添加输出标签函数"""

    print("\n无对应输出标签函数，请添加输出标签函数。")
    return 0


def get_class(class_path):
    """[读取指定的class文件至class_list]

    Parameters
    ----------
    class_path : [str]
        [类别文件路径]

    Returns
    -------
    class_list : [list]
        [读取的类别列表]
    """

    print('collect source ')
    class_list = []
    with open(class_path, 'r') as class_file:
        for one_line in class_file.readlines():
            re_one_line = one_line.replace(' ', '').lower()
            class_list.append(re_one_line.strip('\n'))

    return class_list


def get_names_list_path(input_path):
    """[获取names文件路径]

    Parameters
    ----------
    input_path : [str]
        [源数据集输入路径]

    Returns
    -------
    clss_path : [str]
        [类别文件路径]
    """

    clss_path = ''
    for a in os.listdir(input_path):
        if os.path.splitext(a)[-1] == 'names':
            clss_path = os.path.join(input_path, a)

    return clss_path


def get_fix_class_dict(fix_class_file_path):
    """[按fix_class_file文件对类别进行融合，生成标签融合字典]

    Parameters
    ----------
    fix_class_file_path : [str]
        [数据集标签融合字典文件路径]

    Returns
    -------
    fix_class_dict : [dict]
        [数据集标签融合字典]
    """

    fix_class_dict = {}
    with open(fix_class_file_path, 'r') as class_file:
        for one_line in class_file.readlines():     # 获取class修改文件内容
            one_line_list = one_line.replace(
                '\n', '').rstrip(' ').lower().split(':')
            one_line_list[-1] = one_line_list[-1].split(' ')   #
            # 细分类别插入修改后class类别，类别融合
            one_line_list[-1].insert(0, one_line_list[0])
            for one_class in one_line_list[1]:     # 清理列表中的空值
                if one_class == '':
                    one_line_list[1].pop(one_line_list[1].index(one_class))
            key = one_line_list[0]
            fix_class_dict[key] = one_line_list[1]      # 将新类别添加至融合类别字典

    return fix_class_dict


def check_out_file_exists(output_file_path):
    """[检查路径是否存在，存在则删除已存在路径]

    Parameters
    ----------
    output_file_path : [str]
        [输出路径]

    Returns
    -------
    output_file_path : [str]
        [输出路径]
    """

    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    return output_file_path


def check_in_file_exists(input_file_path):
    """[检查路径是否存在]

    Parameters
    ----------
    input_file_path : [str]
        [输入路径]

    Returns
    -------
    input_file_path : [str]
        [输入路径]
    """

    if os.path.exists(input_file_path):
        return input_file_path
    # assert path was found
    assert os.path.exists(
        input_file_path), 'Input path Not Found: %s' % input_file_path


def get_new_class_names_list(classes_fix_dict):
    """[获取融合后的类别列表]

    Parameters
    ----------
    classes_fix_dict : [dict]
        [数据集标签融合类别字典]

    Returns
    -------
    new_class_names_list : [list]
        [融合后标签类别列表]
    """

    new_class_names_list = []
    for key in classes_fix_dict.keys():     # 依改后类别字典，生成融合后的类别列表
        new_class_names_list.append(key.replace(' ', ''))

    return new_class_names_list


def get_src_total_label_path_list(total_label_list_path, image_type):
    """[获取源数据集全部标签路径列表]

    Parameters
    ----------
    total_label_list_path : [list]
        [源数据集全部标签路径]
    image_type : [str]
        [图片类别]

    Returns
    -------
    src_total_label_path_list : [list]
        [数据集全部标签路径列表]
    """

    src_total_label_path_list = []
    with open(total_label_list_path, 'r') as total:     # 获取数据集全部源标签列表
        for one_line in total.readlines():  # 获取源数据集labels路径列表
            src_total_label_path_list.append(one_line.replace('\n', '')
                                                     .replace('JPEGImages', 'labels')
                                                     .replace('.'+image_type, '.txt'))

    return src_total_label_path_list


def ldp_set_folds_list():
    """[返回输出数据集组织结构列表]

    Returns
    -------
    [list]
        ['Annotations', 'JPEGImages', 'ImageSets', 'labels', 'source_label']
    """

    return ['Annotations', 'JPEGImages', 'ImageSets', 'labels', 'source_label']


def func_None(*args):
    """如无对应model的fold函数，需添加函数"""

    print("\nCannot find function, you shoule appen the function.")
    return 0


set_fold_func_dict = {"ldp": ldp_set_folds_list,
                      "hy": ldp_set_folds_list,
                      "myxb": ldp_set_folds_list,
                      "kitti": ldp_set_folds_list,
                      "pascal": ldp_set_folds_list,
                      "sjt": ldp_set_folds_list}


def pickup_fold_function(model, *args):
    """[根据输入类别挑选输出数据集组织结构]

    Parameters
    ----------
    model : [str]
        [输出数据集组织结构]

    Returns
    -------
    set_fold_func_dict.get() : [function]
        [输出数据集组织结构函数]
    """

    return set_fold_func_dict.get(model, func_None)(*args)  # 返回对应类别更名函数


def get_dataset_scene(dateset_list):
    """[获取数据集场景元组]

    Parameters
    ----------
    dateset_list : [list]
        [数据集annotation提取数据]

    Returns
    -------
    [tuple]
        [数据集按名称分类场景元组]
    """

    class_list = []
    class_set = ()
    for one in dateset_list:
        image_name_list = (one.image_name).split('_')     # 对图片名称进行分段，区分场景
        image_name_str = ''
        for one_name in image_name_list[:-1]:    # 读取切分图片名称的值，去掉编号及后缀
            image_name_str += one_name   # target_image_name_str为图片包含场景的名称
        class_list.append(image_name_str)
    class_set = set(class_list)

    return class_set


def count_free_space_area(point_miny_minx, point_maxy_maxx):
    """[获取空闲空间面积]

    Parameters
    ----------
    point_miny_minx : [list]
        [闲置矩形框左上点]
    point_maxy_maxx : [list]
        [闲置矩形框右下点]

    Returns
    -------
    free_space_area : [float]
        [空闲空间面积]
    """
    free_space_area = 0
    w = point_maxy_maxx[2] - point_miny_minx[2]
    h = point_maxy_maxx[1] - point_miny_minx[1]
    free_space_area = w * h

    return free_space_area

# def max_sum_1(n, a):
#     """[获取最大子序列和]

#     Parameters
#     ----------
#     n : [int]
#         [matrix行数]
#     a : [matrix]
#         [矩阵]

#     Returns
#     -------
#     sum : [int]
#         [获取最大子序列和]
#     """

#     sum = 0
#     b = 0
#     for i in range(0, n):
#         if b > 0:
#             b += a[i]
#         else:
#             b = a[i]
#         if b > sum:
#             sum = b
#     return sum

# def get_largest_submatrix(matrix):
#     """[获取最大子矩阵和]

#     Parameters
#     ----------
#     matrix : [matrix]
#         [matrix]

#     Returns
#     -------
#     free_space_area : [int]
#         [最大子矩阵元素和即面积]
#     """

#     max_free_space_rectangle, free_space_area = 0, 0
#     for i in tqdm(range(0, matrix.shape[0])):
#         b = []
#         for k in tqdm(range(0, matrix.shape[1])):
#             b.append(0)
#         for j in range(i, matrix.shape[0]):
#             for k in range(0, matrix.shape[1]):
#                 b[k] += matrix[j][k]
#             max = max_sum_1(matrix.shape[1], b)
#             if max > free_space_area:
#                 free_space_area = max

#     return free_space_area


def takeSecond(elem):
    """[获取列表的第二个元素]

    Parameters
    ----------
    elem : [elem]
        [列表元素]

    Returns
    -------
    [elem]
        [列表的第二个元素]
    """

    return elem[1]


# RT:RightTop
# LB:LeftBottom
# def IOU(rectangle A, rectangleB):
#     W = min(A.RT.x, B.RT.x) - max(A.LB.x, B.LB.x)
#     H = min(A.RT.y, B.RT.y) - max(A.LB.y, B.LB.y)
#     if W <= 0 or H <= 0:
#         return 0;
#     SA = (A.RT.x - A.LB.x) * (A.RT.y - A.LB.y)
#     SB = (B.RT.x - B.LB.x) * (B.RT.y - B.LB.y)
#     cross = W * H
#     return cross/(SA + SB - cross)

def cheak_total_images_data_list(total_images_data_list):
    """[检查总图片信息列表，若图片无真实框则剔除]

    Args:
        total_images_data_list ([list]): [总图片信息列表]

    Returns:
        [list]: [返回清理无真实框图片后的总图片列表]
    """

    new_total_images_data_list = []
    total = len(total_images_data_list)
    empty = 0
    print("\nStart cheak empty annotation.")
    for one_image_data in tqdm(total_images_data_list):
        if 0 != (len(one_image_data.true_box_list)):
            new_total_images_data_list.append(one_image_data)
        else:
            empty += 1
    total_last = total - empty
    print("\nUsefull images: {n} \nEmpty images: {m}\n".format(
        n=total_last, m=empty))

    return new_total_images_data_list


def delete_empty_images(output_path, total_label_list):
    """[删除无labels的图片]

    Args:
        output_path ([str]): [输出数据集路径]
        total_label_list ([list]): [全部labels列表]
    """

    images_path = os.path.join(output_path, 'JPEGImages')
    total_images_list = os.listdir(images_path)
    labels_to_images_list = []

    for n in total_label_list:
        n += '.jpg'
        labels_to_images_list.append(n)

    need_delete_images_list = [
        y for y in total_images_list if y not in labels_to_images_list]
    print('\nStart delete no ture box images:')
    delete_images = 0
    if 0 != len(need_delete_images_list):
        for delete_one in tqdm(need_delete_images_list):
            delete_image_path = os.path.join(images_path, delete_one)
            if os.path.exists(delete_image_path):
                os.remove(delete_image_path)
                delete_images += 1
    else:
        print("\nNo images need delete.")
    print("\nDelete images: %d" % delete_images)

    return


def delete_empty_ann(output_path, total_label_list):
    """[删除无labels的annotations]

    Args:
        output_path ([str]): [输出数据集路径]
        total_label_list ([list]): [全部labels列表]
    """

    annotations_path = os.path.join(output_path, 'Annotations')
    total_images_list = os.listdir(annotations_path)
    labels_to_annotations_list = []

    for n in total_label_list:
        n += '.xml'
        labels_to_annotations_list.append(n)

    need_delete_annotations_list = [
        y for y in total_images_list if y not in labels_to_annotations_list]
    print('\nStart delete no ture box images:')
    delete_annotations = 0
    if 0 != len(need_delete_annotations_list):
        for delete_one in tqdm(need_delete_annotations_list):
            delete_annotation_path = os.path.join(annotations_path, delete_one)
            if os.path.exists(delete_annotation_path):
                os.remove(delete_annotation_path)
                delete_annotations += 1
    else:
        print("\nNo annotation need delete.")
    print("\nDelete annotations: %d" % delete_annotations)

    return