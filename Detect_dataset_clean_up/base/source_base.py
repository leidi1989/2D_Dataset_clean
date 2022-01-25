'''
Description:
Version:
Author: Leidi
Date: 2021-08-03 22:18:39
LastEditors: Leidi
LastEditTime: 2022-01-25 15:55:02
'''
import os
import shutil
from tqdm import tqdm
import multiprocessing

from utils.image_form_transform import image_transform_function
from base.dataset_characteristic import ANNOTATAION_RENAME_WITH_FOLDER, IMAGE_RENAME_WITH_FOLDER


def source(dataset: dict) -> None:
    """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀，复制源数据集标签文件至暂存数据集]

    Args:
        dataset (dict): [数据集信息字典]
    """
    temp_image(dataset)
    temp_annotation(dataset)


def copy_image(dataset: dict, root: str, n: str) -> None:
    """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀]

    Args:
        dataset (dict): [数据集信息字典]
        root (str): [文件所在目录]
        n (str): [文件名]

    """

    image = os.path.join(root, n)
    if dataset['source_dataset_stype'] in IMAGE_RENAME_WITH_FOLDER:
        folder = root.split(
            os.sep)[IMAGE_RENAME_WITH_FOLDER[dataset['source_dataset_stype']]]
        dataset['file_prefix'] += folder + dataset['prefix_delimiter']
        temp_image = os.path.join(
            dataset['source_images_folder'], dataset['file_prefix'] + n)
    else:
        temp_image = os.path.join(
            dataset['source_images_folder'], dataset['file_prefix'] + n)
    if dataset['source_image_form'] != dataset['target_image_form']:
        dataset['transform_type'] = dataset['source_image_form'] + \
            '_' + dataset['target_image_form']
        image_transform_function(
            dataset['transform_type'], image, temp_image)
        return
    else:
        shutil.copy(image, temp_image)
        return


def copy_annotation(dataset: dict, root: str, n: str) -> None:
    """[复制源数据集标签文件至目标数据集中的source_annotations中]

    Args:
        dataset (dict): [数据集信息字典]
        root (str): [文件所在目录]
        n (str): [文件名]
    """

    annotation = os.path.join(root, n)
    # 按数据集标签保存规则进行标签更名
    if dataset['source_dataset_stype'] in ANNOTATAION_RENAME_WITH_FOLDER:
        folder = root.split(
            os.sep)[ANNOTATAION_RENAME_WITH_FOLDER[dataset['source_dataset_stype']]]
        dataset['file_prefix'] += folder + dataset['prefix_delimiter']
        temp_annotation = os.path.join(
            dataset['source_annotations_folder'], dataset['file_prefix'] + n)
    else:
        temp_annotation = os.path.join(
            dataset['source_annotations_folder'], n)
    shutil.copy(annotation, temp_annotation)

    return


def temp_image(dataset: dict) -> None:
    """[移动源数据集图片至目标数据集，同时进行更名和格式转换]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy images: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if os.path.splitext(n)[-1] == dataset['source_image_form'] \
                or os.path.splitext(n)[-1] == '.png':
                pool.apply_async(copy_image,
                                 args=(dataset, root, n),)
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))
    return


def temp_annotation(dataset: dict) -> None:
    """[移动源数据集标签至目标数据集]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy annotations: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if os.path.splitext(n)[-1] == dataset['source_annotation_form']:
                pool.apply_async(copy_annotation,
                                 args=(dataset, root, n),)
        pool.close()
        pool.join()
    print('Move annotations count: {}\n'.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    return
