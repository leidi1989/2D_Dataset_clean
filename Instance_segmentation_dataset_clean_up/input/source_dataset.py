'''
Description: 
Version: 
Author: Leidi
Date: 2021-11-08 10:30:52
LastEditors: Leidi
LastEditTime: 2021-12-22 14:34:01
'''
import os
from tqdm import tqdm
import multiprocessing

import input.source_dataset_function as F


def apolloscape_lane_segment(dataset: dict) -> None:
    """[拷贝apolloscape_lane_segment数据集中的image及annotation至temp文件夹]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy images: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_image_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_image,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))

    print('Copy annotations: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_annotation_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_annotation,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move annotations count: {}\n'.format(
        len(os.listdir(dataset['source_annotations_folder']))))

    return


def huaweiyun_segment(dataset: dict) -> None:
    """[拷贝huawei_segment数据集中的image及annotation至temp文件夹]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy images: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_image_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_image,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))

    print('Copy annotations: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_annotation_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_annotation,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move annotations count: {}\n'.format(
        len(os.listdir(dataset['source_annotations_folder']))))

    return


def coco2017(dataset: dict) -> None:
    """[拷贝coco2017数据集中的image及annotation至temp文件夹]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy images: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_image_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_image,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))

    print('Copy annotations: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_annotation_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_annotation,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move annotations count: {}\n'.format(
        len(os.listdir(dataset['source_annotations_folder']))))

    return


def cityscapes(dataset: dict) -> None:
    """[拷贝cityscapes数据集中的image及annotation至temp文件夹]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy images: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_image_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_image,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))

    print('Copy annotations: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_annotation_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_annotation,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move annotations count: {}\n'.format(
        len(os.listdir(dataset['source_annotations_folder']))))

    return


def bdd100k(dataset: dict) -> None:
    """[拷贝bdd100k数据集中的image及annotation至temp文件夹]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy images: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_image_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_image,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))

    print('Copy annotations: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_annotation_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_annotation,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move annotations count: {}\n'.format(
        len(os.listdir(dataset['source_annotations_folder']))))

    return


def cvat_image_1_1(dataset: dict) -> None:
    """[拷贝cvat_image_1_1数据集中的image及annotation至temp文件夹]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy images: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_image_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_image,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))

    print('Copy annotations: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        try:
            prefix = root.split('-')[1].replace('_', '')
        except:
            prefix = ''
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_annotation_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_annotation,
                                 args=(dataset, root, n, prefix,))
        pool.close()
        pool.join()
    print('Move annotations count: {}\n'.format(
        len(os.listdir(dataset['source_annotations_folder']))))

    return


def hy_val(dataset: dict) -> None:
    """[拷贝hy_val数据集中的image及annotation至temp文件夹]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy images: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_image_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_image,
                                 args=(dataset, root, n))
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))

    print('Start create fake json:')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_annotation,
                             args=(dataset, root, n))
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))

    return


def yunce_segment_coco(dataset: dict) -> None:
    """[拷贝huawei_segment数据集中的image及annotation至temp文件夹]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy images: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_image_form']) or n.endswith('png'):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_image,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))

    print('Copy annotations: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_annotation_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_annotation,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move annotations count: {}\n'.format(
        len(os.listdir(dataset['source_annotations_folder']))))

    return


def yunce_segment_coco_one_image(dataset: dict) -> None:
    """[拷贝huawei_segment数据集中的image及annotation至temp文件夹]

    Args:
        dataset (dict): [数据集信息字典]
    """

    print('Copy images: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_image_form']) or n.endswith('png'):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_image,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move images count: {}\n'.format(
        len(os.listdir(dataset['source_images_folder']))))

    print('Copy annotations: ')
    for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
        pool = multiprocessing.Pool(dataset['workers'])
        for n in tqdm(files):
            if n.endswith(dataset['source_annotation_form']):
                pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_annotation,
                                 args=(dataset, root, n,))
        pool.close()
        pool.join()
    print('Move annotations count: {}\n'.format(
        len(os.listdir(dataset['source_annotations_folder']))))

    return
