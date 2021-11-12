'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-11 03:28:09
LastEditors: Leidi
LastEditTime: 2021-10-21 20:07:32
'''
from base.image_base import *
from utils.utils import check_output_path

import os
import shutil
from tqdm import tqdm


def PASCAL_VOC_FRAMEWORK(dataset) -> None:

    # 调整image
    print('Change images folder.')
    image_output_path = check_output_path(
        os.path.join(dataset['target_path'], 'JPEGImages'))
    os.rename(dataset['source_images_folder'], image_output_path)

    # 调整ImageSets
    print('Change ImageSets folder:')
    imagesets_path = check_output_path(
        os.path.join(dataset['target_path'], 'Dataset_infomations'))
    for n in tqdm(dataset['temp_divide_file_list']):
        print('Update {}'.format(n.split(os.sep)[-1]))
        output_list = []
        with open(n, 'r') as f:
            annotation_path_list = f.read().splitlines()
            for m in annotation_path_list:
                output_list.append(m.replace('source_images', 'JPEGImages'))
            f.close()

        with open(os.path.join(imagesets_path, n.split(os.sep)[-1]), 'w') as f:
            for m in output_list:
                f.write('%s\n' % m)
            f.close()
        os.remove(n)

    # 调整文件夹
    print('Update folder.')
    shutil.move(os.path.join(dataset['temp_informations_folder'],
                             'Main'), os.path.join(imagesets_path, 'Main'))
    shutil.rmtree(dataset['temp_annotation_folder'])
    os.rename(dataset['temp_information_folder'], os.path.join(
        dataset['output_path'], 'Dataset_information'))

    return


def COCO_2017_FRAMEWORK(dataset):

    # 调整image
    print('Change images folder.')
    image_output_path = check_output_path(
        os.path.join(dataset['output_path'], 'images'))
    os.rename(dataset['temp_images_folder'], image_output_path)

    # 调整ImageSets
    print('Change ImageSets folder:')
    imagesets_path = check_output_path(
        os.path.join(dataset['output_path'], 'ImageSets'))
    for n in dataset['temp_divide_file_list']:
        print('Update {} '.format(n.split(os.sep)[-1]))
        output_list = []
        with open(n, 'r') as f:
            annotation_path_list = f.read().splitlines()
            for m in tqdm(annotation_path_list):
                output_list.append(m.replace('temp_images', 'images'))
            f.close()

        with open(os.path.join(imagesets_path, n.split(os.sep)[-1]), 'w') as f:
            for m in output_list:
                f.write('%s\n' % m)
            f.close()
        os.remove(n)

    # 调整文件夹
    print('Update folder.')
    shutil.rmtree(os.path.join(dataset['temp_information_folder'], 'Main'))
    os.rename(dataset['temp_information_folder'], os.path.join(
        dataset['output_path'], 'Dataset_information'))
    shutil.rmtree(dataset['temp_annotation_folder'])

    return


def YOLO_FRAMEWORK(dataset):

    # 调整image
    print('Change images folder.')
    image_output_path = check_output_path(
        os.path.join(dataset['output_path'], 'images'))
    os.rename(dataset['temp_images_folder'], image_output_path)

    # 调整ImageSets
    print('Change ImageSets folder:')
    imagesets_path = check_output_path(
        os.path.join(dataset['output_path'], 'ImageSets'))
    for n in dataset['temp_divide_file_list']:
        print('Update {} '.format(n.split(os.sep)[-1]))
        output_list = []
        with open(n, 'r') as f:
            annotation_path_list = f.read().splitlines()
            for m in tqdm(annotation_path_list):
                output_list.append(m.replace('temp_images', 'images'))
            f.close()

        with open(os.path.join(imagesets_path, n.split(os.sep)[-1]), 'w') as f:
            for m in output_list:
                f.write('%s\n' % m)
            f.close()
        os.remove(n)

    # 调整文件夹
    print('Update folder.')
    shutil.rmtree(os.path.join(dataset['temp_information_folder'], 'Main'))
    os.rename(dataset['temp_information_folder'], os.path.join(
        dataset['output_path'], 'Dataset_information'))
    shutil.rmtree(dataset['temp_annotation_folder'])

    return


framework_function_dict = {'pascal_voc': PASCAL_VOC_FRAMEWORK,
                           'coco2017': COCO_2017_FRAMEWORK,
                           'yolo': YOLO_FRAMEWORK
                           }


def framework_funciton(dataset_stype: str, *args):
    """[获取指定类别数据集annotation输出函数。]

    Args:
        dataset_style (str): [输出数据集类别。]

    Returns:
        [function]: [返回指定类别数据集输出函数。]
    """
    # try:
    return framework_function_dict.get(dataset_stype)(*args)
    # except:
    #     print("Annotation output fail, need update {} annotation output function！".format(
    #         dataset_stype))
