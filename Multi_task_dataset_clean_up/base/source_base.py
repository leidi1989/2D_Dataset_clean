'''
Description:
Version:
Author: Leidi
Date: 2021-08-03 22:18:39
LastEditors: Leidi
LastEditTime: 2021-10-22 23:23:52
'''
import os
import shutil
from tqdm import tqdm
import multiprocessing

from utils.utils import create_fake_annotation
from utils.image_form_transform import image_transform_function


def source(dataset: dict) -> None:
    """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀，复制源数据集标签文件至暂存数据集]

    Args:
        dataset (dict): [数据集信息字典]
    """

    temp_image(dataset)
    if dataset['source_dataset_stype'] == 'hy_val':
        create_fake_annotation(dataset)
        return

    temp_annotation(dataset)

    return


def copy_image(dataset: dict, root: str, n: str) -> None:
    """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀]

    Args:
        dataset (dict): [数据集信息字典]
        root (str): [文件所在目录]
        n (str): [文件名]

    """

    image = os.path.join(root, n)
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
    temp_annotation = os.path.join(
        dataset['source_annotations_folder'], n)
    shutil.copy(annotation, temp_annotation)

    return


def temp_image(dataset: dict) -> None:
    """[移动源数据集图片至目标数据集，同时进行更名和格式转换]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('\nCopy images: ')
    for root, dirs, files in os.walk(dataset['source_path']):
        with tqdm(total=len(files)) as t:
            pool = multiprocessing.Pool(dataset['workers'])
            for n in tqdm(files):
                if n.endswith(dataset['source_image_form']):
                    pool.apply_async(copy_image,
                                     args=(dataset, root, n),
                                     callback=t.update())
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
    for root, dirs, files in os.walk(dataset['source_path']):
        with tqdm(total=len(files)) as t:
            pool = multiprocessing.Pool(dataset['workers'])
            for n in tqdm(files):
                if n.endswith(dataset['source_annotation_form']):
                    pool.apply_async(copy_annotation,
                                     args=(dataset, root, n),
                                     callback=t.update())
            pool.close()
            pool.join()
    print('Move annotations count: {}\n'.format(
        len(os.listdir(dataset['source_annotations_folder']))))

    return
