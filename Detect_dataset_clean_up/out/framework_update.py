from base.image_base import *
from utils.utils import check_output_path

import os
import shutil
from tqdm import tqdm
import multiprocessing

from utils.utils import err_call_back
from out.framwork_update_function import yolo, pascal_voc, coco2017


def PASCAL_VOC_FRAMEWORK(dataset) -> None:
    """[输出PASCAL VOC格式数据集]

    Args:
        dataset (dict): [数据集信息字典]
    """

    dataset_output_path = check_output_path(
        os.path.join(dataset['target_path'], 'PASCAL_VOC'))
    images_output_path = check_output_path(
        os.path.join(dataset_output_path, 'JPEGImages'))
    annotations_output_path = check_output_path(
        os.path.join(dataset_output_path, 'Annotations'))
    imagesets_output_path = check_output_path(
        os.path.join(dataset_output_path, 'ImageSets'))

    image_count = 0
    annotation_count = 0
    # 调整ImageSets
    print('Create output dataset:')
    for temp_divide_file in tqdm(dataset['temp_divide_file_list'][1:4]):
        file_name_list = []
        with open(temp_divide_file, 'r') as f:
            images_path_list = f.read().splitlines()
            # 创建ImageSets中数据集分割文件
            for image_path in tqdm(images_path_list):
                file_name_list.append(os.path.splitext(image_path.split(os.sep)[-1])[0])

        pool = multiprocessing.Pool(dataset['workers'])
        process_output = multiprocessing.Manager().dict({"image_count": 0,
                                                         "annotation_count": 0,
                                                         })
        for file_name in tqdm(file_name_list):
            pool.apply_async(func=pascal_voc.copy_images_annotations,
                             args=(dataset, file_name,
                                   process_output, images_output_path, annotations_output_path),
                             error_callback=err_call_back)
        pool.close()
        pool.join()

        with open(os.path.join(imagesets_output_path, temp_divide_file.split(os.sep)[-1]), 'w') as s:
            for file_name in file_name_list:
                s.write('%s\n' % os.path.join(images_output_path, file_name))
            s.close()

        image_count += process_output['image_count']
        annotation_count += process_output['annotation_count']

    print('Copy images total: {}'.format(image_count))
    print('Copy annotations total: {}'.format(annotation_count))

    return


def COCO2017_FRAMEWORK(dataset):
    """[输出COCO2017格式数据集]

    Args:
        dataset (dict): [数据集信息字典]
    """

    dataset_output_path = check_output_path(
        os.path.join(dataset['target_path'], 'COCO2017'))
    train_images_output_path = check_output_path(
        os.path.join(dataset_output_path, 'train'))
    test_images_output_path = check_output_path(
        os.path.join(dataset_output_path, 'test'))
    val_images_output_path = check_output_path(
        os.path.join(dataset_output_path, 'val'))
    annotations_output_path = check_output_path(
        os.path.join(dataset_output_path, 'annotations'))

    images_output_path_list = [train_images_output_path,
                               test_images_output_path, val_images_output_path]

    image_count = 0
    annotation_count = 0
    # 调整ImageSets
    print('Create output dataset:')
    for temp_divide_file, images_output_path in zip(dataset['temp_divide_file_list'][1:4], images_output_path_list):
        file_name_list = []
        with open(temp_divide_file, 'r') as f:
            images_path_list = f.read().splitlines()
            # 创建ImageSets中数据集分割文件
            for image_path in tqdm(images_path_list):
                file_name_list.append(os.path.splitext(image_path.split(os.sep)[-1])[0])

        pool = multiprocessing.Pool(dataset['workers'])
        process_output = multiprocessing.Manager().dict({"image_count": 0,
                                                         })
        for file_name in tqdm(file_name_list):
            pool.apply_async(func=coco2017.copy_images,
                             args=(dataset, file_name,
                                   process_output, images_output_path),
                             error_callback=err_call_back)
        pool.close()
        pool.join()

        image_count += process_output['image_count']

    for annotation_name in os.listdir(dataset['target_annotations_folder']):
        annotation_path = os.path.join(
            dataset['target_annotations_folder'], annotation_name)
        annotation_output_path = os.path.join(
            annotations_output_path, annotation_name)
        shutil.copy(annotation_path, annotation_output_path)
        annotation_count += 1

    print('Copy images total: {}'.format(image_count))
    print('Copy annotations total: {}'.format(annotation_count))

    return


def YOLO_FRAMEWORK(dataset: dict) -> None:
    """[输出YOLO格式数据集]

    Args:
        dataset (dict): [数据集信息字典]
    """

    dataset_output_path = check_output_path(
        os.path.join(dataset['target_path'], 'YOLO'))
    images_output_path = check_output_path(
        os.path.join(dataset_output_path, 'images'))
    annotations_output_path = check_output_path(
        os.path.join(dataset_output_path, 'labels'))
    imagesets_output_path = check_output_path(
        os.path.join(dataset_output_path, 'ImageSets'))

    image_count = 0
    annotation_count = 0
    # 调整ImageSets
    print('Create output dataset:')
    for temp_divide_file in dataset['temp_divide_file_list'][1:4]:
        file_name_list = []
        with open(temp_divide_file, 'r') as f:
            images_path_list = f.read().splitlines()
            # 创建ImageSets中数据集分割文件
            for image_path in tqdm(images_path_list):
                file_name_list.append(os.path.splitext(image_path.split(os.sep)[-1])[0])

        pool = multiprocessing.Pool(dataset['workers'])
        process_output = multiprocessing.Manager().dict({"image_count": 0,
                                                         "annotation_count": 0,
                                                         })
        for file_name in tqdm(file_name_list):
            pool.apply_async(func=yolo.copy_images_annotations,
                             args=(dataset, file_name,
                                   process_output, images_output_path, annotations_output_path),
                             error_callback=err_call_back)
        pool.close()
        pool.join()

        with open(os.path.join(imagesets_output_path, temp_divide_file.split(os.sep)[-1]), 'w') as s:
            for file_name in file_name_list:
                s.write('%s\n' % os.path.join(images_output_path, file_name))
            s.close()

        image_count += process_output['image_count']
        annotation_count += process_output['annotation_count']

    print('Copy images total: {}'.format(image_count))
    print('Copy annotations total: {}'.format(annotation_count))

    return


framework_function_dict = {'pascal_voc': PASCAL_VOC_FRAMEWORK,
                           'coco2017': COCO2017_FRAMEWORK,
                           'yolo': YOLO_FRAMEWORK
                           }


def framework_funciton(dataset_stype: str, *args):
    """[获取指定类别数据集annotation输出函数。]

    Args:
        dataset_style (str): [输出数据集类别。]

    Returns:
        [function]: [返回指定类别数据集输出函数。]
    """

    return framework_function_dict.get(dataset_stype)(*args)
