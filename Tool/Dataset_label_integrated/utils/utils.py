# -*- coding: utf-8 -*-
import os


def check_input_path(path):
    """检查输入路径是否存在"""
    
    if os.path.exists(path):
        return path
    else:
        assert os.path.exists(path), 'Input path Not Found: %s' % path    # assert path was found


def check_output_path(path, attach = ''):
    """检查输出路径是否存在"""

    if os.path.exists(os.path.join(path, attach)):
        return os.path.join(path, attach)
    else:
        print('Creat output path: %s\n' % str(os.path.join(path, attach)))
        os.makedirs(os.path.join(path, attach))
        return os.path.join(path,attach)


def check_image_type(input_label_style):
    """确定输入图片格式"""

    jpg_list = ['ldp', 'hy', 'voc', 'coco', 'pascal']
    png_list = ['hy_highway', 'kitti']

    if not (input_label_style in jpg_list,
            input_label_style in png_list):      # 判断输入的类型在不在已辨认列表中
            print("\n无对应输出图像格式，请添加输出图像格式")
    
    if input_label_style in jpg_list:
        file_type = 'jpg'
    if input_label_style in png_list:
        file_type = 'png'

    return file_type


def check_src_lab_type(input_label_style):
    """确定输入label格式"""

    json_list = ['hy', 'coco']
    xml_list = ['ldp', 'hy_highway', 'pascal']
    txt_list = ['kitti']

    if not (input_label_style in json_list,
            input_label_style in xml_list,
            input_label_style in txt_list):      # 判断输入的类型在不在已辨认列表中
            print("\n无对应输出label格式，请添加输出label格式")
    
    if input_label_style in json_list:
        file_type = 'json'
    if input_label_style in xml_list:
        file_type = 'xml'
    if input_label_style in txt_list:
        file_type = 'txt'

    return file_type


def check_pref(pref):
    """检查是否添加前缀"""

    if pref == '':
        name_pre = pref
    else:
        name_pre = pref + '_'

        return name_pre


def out_func_None(*args):
    """如无对应输出标签函数，提示用户添加输出标签函数"""
    
    print("\n无对应输出标签函数，请添加输出标签函数。")
    return 0


def get_class(class_path):
    """读取指定的class文件至class_list"""

    print('collect source ')
    class_list = []
    with open(class_path, 'r') as class_file:
        for one_line in class_file.readlines():
            re_one_line = one_line.replace(' ', '').lower()
            class_list.append(re_one_line.strip('\n'))

    return class_list


def get_names_list_path(input_path):
    """获取names文件路径"""

    clss_path = ''
    for a in os.listdir(input_path):
        if os.path.splitext(a)[-1]=='names':
            clss_path = os.path.join(input_path, a)
    
    return clss_path


def get_fix_class_dict(fix_class_file_path):
    """按fix_class_file文件对类别进行融合"""

    fix_class_dict = {}
    with open(fix_class_file_path, 'r') as class_file:
        for one_line in class_file.readlines():     # 获取class修改文件内容
            one_line_list = one_line.replace('\n', '').rstrip(' ').lower().split(':')
            one_line_list[-1] = one_line_list[-1].split(' ')   # 
            one_line_list[-1].insert(0, one_line_list[0])   # 细分类别插入修改后class类别，类别融合
            for one_class in one_line_list[1]:     # 清理列表中的空值
                if one_class == '':
                    one_line_list[1].pop(one_line_list[1].index(one_class))
            key = one_line_list[0]
            fix_class_dict[key] = one_line_list[1]      # 将新类别添加至融合类别字典

    return fix_class_dict


def check_out_file_exists(output_file_path):
    """检查路径是否存在，存在则删除已存在路径"""

    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    return output_file_path


def check_in_file_exists(input_file_path):
    """检查路径是否存在"""

    if os.path.exists(input_file_path):
        return input_file_path
    assert os.path.exists(input_file_path), 'Input path Not Found: %s' % input_file_path    # assert path was found


def get_new_class_names_list(classes_fix_dict):
    """获取融合后的类别列表"""

    new_class_names_list = []
    for key in classes_fix_dict.keys():     # 依改后类别字典，生成融合后的类别列表
        new_class_names_list.append(key.replace(' ', ''))

    return new_class_names_list


def get_src_total_label_path_list(total_label_list_path, image_type):
    """获取源数据集全部标签路径列表"""

    src_total_label_path_list = []
    with open(total_label_list_path, 'r') as total:     # 获取数据集全部源标签列表
        for one_line in total.readlines():  # 获取源数据集labels路径列表
            src_total_label_path_list.append(one_line.replace('\n', '')
                                                     .replace('images', 'labels')
                                                     .replace('.'+image_type, '.txt'))
    
    return src_total_label_path_list


def ldp_set_folds_list():
    return ['Annotations', 'images', 'ImageSets', 'labels', 'source label']


def func_None(*args):
    """如无对应model的fold函数，需添加函数"""

    print("\nCannot find function, you shoule appen the function.")
    return 0


set_fold_func_dict = {"ldp":ldp_set_folds_list, 
                      "hy":ldp_set_folds_list, 
                      "kitti":ldp_set_folds_list, 
                      "pascal":ldp_set_folds_list}


def pickup_fold_function(model, *args):
    """根据输入类别挑选更换图片文件名"""

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
