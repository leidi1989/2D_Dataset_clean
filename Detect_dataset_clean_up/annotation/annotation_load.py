'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:43:21
LastEditors: Leidi
LastEditTime: 2022-01-26 09:27:45
'''
import os
import csv
import json
import multiprocessing
from subprocess import call

from sqlalchemy import desc

from utils.utils import *
from annotation.dataset_load_function import yolo, pascal_voc, coco2017, kitti, tt100k, kitti, cctsdb, lisa, \
    sjt, myxb, hy_highway, cvat_coco2017, huawei_segment


def YOLO_LOAD(dataset: dict):
    """[YOLO数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    pool = multiprocessing.Pool(dataset['workers'])
    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({'success_count': 0,
                                                     'fail_count': 0,
                                                     'no_true_box_count': 0,
                                                     'temp_file_name_list': process_temp_file_name_list
                                                     })
    for source_annotations_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        pool.apply_async(func=yolo.load_annotation,
                         args=(dataset, source_annotations_name,
                               process_output,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(
        process_output['fail_count']))
    print('No true box delete images: \t {} '.format(
        process_output['no_true_box_count']))
    print('Convert success:           \t {} '.format(
        process_output['success_count']))
    dataset['temp_file_name_list'] = [
        x for x in process_output['temp_file_name_list']]
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n) for n in dataset['class_list_new']))
        f.close()

    return


def PASCAL_VOC_LOAD(dataset: dict) -> None:
    """[PASCAL VOC数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    pool = multiprocessing.Pool(dataset['workers'])
    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({'success_count': 0,
                                                     'fail_count': 0,
                                                     'no_true_box_count': 0,
                                                     'temp_file_name_list': process_temp_file_name_list
                                                     })
    for source_annotations_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        pool.apply_async(func=pascal_voc.load_annotation,
                         args=(dataset, source_annotations_name,
                               process_output,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(
        process_output['fail_count']))
    print('No true box delete images: \t {} '.format(
        process_output['no_true_box_count']))
    print('Convert success:           \t {} '.format(
        process_output['success_count']))
    dataset['temp_file_name_list'] = [
        x for x in process_output['temp_file_name_list']]
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n) for n in dataset['class_list_new']))
        f.close()

    return


def COCO2017_LOAD(dataset) -> None:
    """[COCO2017数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_true_box_count = 0
    temp_file_name_list = []

    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        source_annotation_path = os.path.join(
            dataset['source_annotations_folder'], source_annotation_name)
        with open(source_annotation_path, 'r') as f:
            data = json.loads(f.read())
        class_dict = {}
        for n in data['categories']:
            class_dict['%s' % n['id']] = n['name']

        # 获取data字典中images内的图片信息，file_name、height、width
        pabr, update = multiprocessing_list_tqdm(data['images'], desc='load image base information')
        total_annotations_dict = multiprocessing.Manager().dict()
        pool = multiprocessing.Pool(dataset['workers'])
        for image_base_information in data['images']:
            pool.apply_async(func=coco2017.load_image_base_information, args=(
                dataset, image_base_information, total_annotations_dict,),
                callback=update,
                error_callback=err_call_back)
        pool.close()
        pool.join()
        pabr.close()

        # 读取目标标注信息
        pabr, update = multiprocessing_list_tqdm(data['annotations'], desc='load image annotation')
        total_image_box_list = []
        pool = multiprocessing.Pool(dataset['workers'])
        for one_annotation in data['annotations']:
            total_image_box_list.append(pool.apply_async(func=coco2017.load_image_annotation, args=(
                dataset, one_annotation, class_dict, total_annotations_dict,),
                callback=update,
                error_callback=err_call_back))
        pool.close()
        pool.join()
        pabr.close()

        total_images_data_dict = {}
        for image_true_box in tqdm(total_image_box_list, desc='total_image_box_list'):
            if image_true_box.get() is None or image_true_box.get()[1] is None:
                continue
            if image_true_box.get()[0] not in total_images_data_dict:
                total_images_data_dict[image_true_box.get(
                )[0]] = total_annotations_dict[image_true_box.get()[0]]
                total_images_data_dict[image_true_box.get()[0]].true_box_list.extend(
                    image_true_box.get()[1])
            else:
                total_images_data_dict[image_true_box.get()[0]].true_box_list.extend(
                    image_true_box.get()[1])

        # 输出读取的source annotation至temp annotation
        pabr, update = multiprocessing_list_tqdm(total_images_data_dict, desc='output temp annotation')
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                         "fail_count": 0,
                                                         "no_true_box_count": 0,
                                                         "temp_file_name_list": process_temp_file_name_list
                                                         })
        pool = multiprocessing.Pool(dataset['workers'])
        for _, image in total_images_data_dict.items():
            pool.apply_async(func=coco2017.output_temp_annotation, args=(
                dataset, image, process_output,),
                callback=update,
                error_callback=err_call_back)
        pool.close()
        pool.join()
        pabr.close()

        # 更新输出统计
        success_count += process_output['success_count']
        fail_count += process_output['fail_count']
        no_true_box_count += process_output['no_true_box_count']
        temp_file_name_list += process_output['temp_file_name_list']

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No segmentation delete images: \t {} '.format(no_true_box_count))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n)
                              for n in dataset['class_list_new']))
        f.close()

    return


def HUAWEI_SEGMENT_LOAD(dataset) -> None:
    """[HUAWEI_SEGMENT数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_true_box_count = 0
    temp_file_name_list = []

    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        source_annotation_path = os.path.join(
            dataset['source_annotations_folder'], source_annotation_name)
        with open(source_annotation_path, 'r') as f:
            data = json.loads(f.read())
        class_dict = {}
        for n in data['categories']:
            class_dict['%s' % n['id']] = n['name']

        # 获取data字典中images内的图片信息，file_name、height、width
        total_annotations_dict = multiprocessing.Manager().dict()
        pool = multiprocessing.Pool(dataset['workers'])
        for image_base_information in tqdm(data['images']):
            pool.apply_async(func=huawei_segment.load_image_base_information, args=(
                dataset, image_base_information, total_annotations_dict,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 读取目标标注信息
        total_image_box_list = []
        pool = multiprocessing.Pool(dataset['workers'])
        for one_annotation in tqdm(data['annotations']):
            total_image_box_list.append(pool.apply_async(func=huawei_segment.load_image_annotation, args=(
                dataset, one_annotation, class_dict, total_annotations_dict,),
                error_callback=err_call_back))
        pool.close()
        pool.join()

        total_images_data_dict = {}
        for image_true_box in total_image_box_list:
            if image_true_box.get() is None:
                continue
            if image_true_box.get()[0] not in total_images_data_dict:
                total_images_data_dict[image_true_box.get(
                )[0]] = total_annotations_dict[image_true_box.get()[0]]
                total_images_data_dict[image_true_box.get()[0]].true_box_list.extend(
                    image_true_box.get()[1])
            else:
                total_images_data_dict[image_true_box.get()[0]].true_box_list.extend(
                    image_true_box.get()[1])

        # 输出读取的source annotation至temp annotation
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                         "fail_count": 0,
                                                         "no_true_box_count": 0,
                                                         "temp_file_name_list": process_temp_file_name_list
                                                         })
        pool = multiprocessing.Pool(dataset['workers'])
        for _, image in tqdm(total_images_data_dict.items()):
            pool.apply_async(func=huawei_segment.output_temp_annotation, args=(
                dataset, image, process_output,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 更新输出统计
        success_count += process_output['success_count']
        fail_count += process_output['fail_count']
        no_true_box_count += process_output['no_true_box_count']
        temp_file_name_list += process_output['temp_file_name_list']

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No segmentation delete images: \t {} '.format(no_true_box_count))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n)
                              for n in dataset['class_list_new']))
        f.close()

    return


def TT100K_LOAD(dataset: dict) -> None:
    """[蚂蚁雄兵数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_true_box_count = 0
    temp_file_name_list = []

    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        source_annotation_path = os.path.join(
            dataset['source_annotations_folder'], source_annotation_name)
        with open(source_annotation_path, 'r') as f:
            data = json.loads(f.read())
            # 获取data字典中images内的图片信息，file_name、height、width
            process_temp_file_name_list = multiprocessing.Manager().list()
            process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                             "fail_count": 0,
                                                             "no_true_box_count": 0,
                                                             "temp_file_name_list": process_temp_file_name_list
                                                             })
            pool = multiprocessing.Pool(dataset['workers'])
            for image_id, out_image in tqdm(data['imgs'].items()):
                pool.apply_async(func=tt100k.load_annotation, args=(
                    dataset, image_id, out_image, process_output,),
                    error_callback=err_call_back)
            pool.close()
            pool.join()

            success_count += process_output['success_count']
            fail_count += process_output['fail_count']
            no_true_box_count += process_output['no_true_box_count']
            temp_file_name_list += [x for x in process_output['temp_file_name_list']]
            f.close()

    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No true box delete images: \t {} '.format(no_true_box_count))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n) for n in dataset['class_list_new']))
        f.close()

    return


def MYXB_LOAD(dataset: dict) -> None:
    """[蚂蚁雄兵数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_true_box_count = 0
    temp_file_name_list = []

    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        source_annotation_path = os.path.join(
            dataset['source_annotations_folder'], source_annotation_name)
        with open(source_annotation_path, 'r', encoding='utf-8') as f:
            total_data = json.load(f)
            pool = multiprocessing.Pool(dataset['workers'])
            process_temp_file_name_list = multiprocessing.Manager().list()
            process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                             "fail_count": 0,
                                                             "no_true_box_count": 0,
                                                             "temp_file_name_list": process_temp_file_name_list
                                                             })
            for one_data in total_data:
                pool.apply_async(func=myxb.load_annotation,
                                 args=(dataset, one_data,
                                       process_output,),
                                 error_callback=err_call_back)
            pool.close()
            pool.join()

            success_count += process_output['success_count']
            fail_count += process_output['fail_count']
            no_true_box_count += process_output['no_true_box_count']
            temp_file_name_list += [x for x in process_output['temp_file_name_list']]
            f.close()

    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No true box delete images: \t {} '.format(no_true_box_count))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n) for n in dataset['class_list_new']))
        f.close()

    return


def HY_DATASET_LOAD(dataset: dict) -> None:
    """[环宇自定义数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_true_box_count = 0
    temp_file_name_list = []

    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        source_annotation_path = os.path.join(
            dataset['source_annotations_folder'], source_annotation_name)
        with open(source_annotation_path, 'r', encoding='utf-8') as f:
            total_data = json.load(f)
            pool = multiprocessing.Pool(dataset['workers'])
            process_temp_file_name_list = multiprocessing.Manager().list()
            process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                             "fail_count": 0,
                                                             "no_true_box_count": 0,
                                                             "temp_file_name_list": process_temp_file_name_list
                                                             })
            for one_data in total_data:
                pool.apply_async(func=myxb.load_annotation,
                                 args=(dataset, one_data,
                                       process_output,),
                                 error_callback=err_call_back)
            pool.close()
            pool.join()
            success_count += process_output['success_count']
            fail_count += process_output['fail_count']
            no_true_box_count += process_output['no_true_box_count']
            temp_file_name_list += [x for x in process_output['temp_file_name_list']]
            f.close()

    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No true box delete images: \t {} '.format(no_true_box_count))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n) for n in dataset['class_list_new']))
        f.close()

    return


def HY_HIGHWAY_LOAD(dataset: dict) -> None:
    """[环宇高速数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    pool = multiprocessing.Pool(dataset['workers'])
    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                     "fail_count": 0,
                                                     "no_true_box_count": 0,
                                                     "temp_file_name_list": process_temp_file_name_list
                                                     })
    for source_annotations_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        pool.apply_async(func=hy_highway.load_annotation,
                         args=(dataset, source_annotations_name,
                               process_output,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(
        process_output['fail_count']))
    print('No true box delete images: \t {} '.format(
        process_output['no_true_box_count']))
    print('Convert success:           \t {} '.format(
        process_output['success_count']))
    dataset['temp_file_name_list'] = [
        x for x in process_output['temp_file_name_list']]
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n) for n in dataset['class_list_new']))
        f.close()

    return


def KITTI_LOAD(dataset: dict) -> None:
    """[KITTI数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    pool = multiprocessing.Pool(dataset['workers'])
    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                     "fail_count": 0,
                                                     "no_true_box_count": 0,
                                                     "temp_file_name_list": process_temp_file_name_list
                                                     })
    for source_annotations_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        pool.apply_async(func=kitti.load_annotation,
                         args=(dataset, source_annotations_name,
                               process_output,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(
        process_output['fail_count']))
    print('No true box delete images: \t {} '.format(
        process_output['no_true_box_count']))
    print('Convert success:           \t {} '.format(
        process_output['success_count']))
    dataset['temp_file_name_list'] = [
        x for x in process_output['temp_file_name_list']]
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n) for n in dataset['class_list_new']))
        f.close()

    return


def CCTSDB_LOAD(dataset: dict) -> None:
    """[CCTSDB数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_true_box_count = 0
    temp_file_name_list = []

    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        source_annotation_path = os.path.join(
            dataset['source_annotations_folder'], source_annotation_name)
        total_annotations_dict = multiprocessing.Manager().dict()
        with open(source_annotation_path, 'r') as f:
            pool = multiprocessing.Pool(dataset['workers'])
            for image_base_information in tqdm(f.read().splitlines()):
                pool.apply_async(func=cctsdb.load_image_base_information, args=(
                    dataset, image_base_information, total_annotations_dict,),
                    error_callback=err_call_back)
            pool.close()
            pool.join()
            f.close()

        # 读取目标标注信息
        total_image_box_list = []
        with open(source_annotation_path, 'r') as f:
            pool = multiprocessing.Pool(dataset['workers'])
            for image_base_information in tqdm(f.read().splitlines()):
                total_image_box_list.append(pool.apply_async(func=cctsdb.load_image_annotation, args=(
                    dataset, image_base_information, total_annotations_dict,),
                    error_callback=err_call_back))
            pool.close()
            pool.join()
            f.close()

        total_images_data_dict = {}
        for image_true_box in total_image_box_list:
            if image_true_box.get() is None:
                continue
            if image_true_box.get()[0] not in total_images_data_dict:
                total_images_data_dict[image_true_box.get(
                )[0]] = total_annotations_dict[image_true_box.get()[0]]
                total_images_data_dict[image_true_box.get()[0]].true_box_list.extend(
                    image_true_box.get()[1])
            else:
                total_images_data_dict[image_true_box.get()[0]].true_box_list.extend(
                    image_true_box.get()[1])

        # 输出读取的source annotation至temp annotation
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                         "fail_count": 0,
                                                         "no_true_box_count": 0,
                                                         "temp_file_name_list": process_temp_file_name_list
                                                         })
        pool = multiprocessing.Pool(dataset['workers'])
        for _, image in tqdm(total_images_data_dict.items()):
            pool.apply_async(func=cctsdb.output_temp_annotation, args=(
                dataset, image, process_output,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 更新输出统计
        success_count += process_output['success_count']
        fail_count += process_output['fail_count']
        no_true_box_count += process_output['no_true_box_count']
        temp_file_name_list += process_output['temp_file_name_list']

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No segmentation delete images: \t {} '.format(no_true_box_count))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n)
                              for n in dataset['class_list_new']))
        f.close()

    return


def LISA_LOAD(dataset: dict) -> None:
    """[LISA数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_true_box_count = 0
    temp_file_name_list = []

    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        if source_annotation_name.split('_')[-1] == 'frameAnnotationsBULB.csv':
            continue
        # if source_annotation_name.split('_')[-1] == 'frameAnnotationsBOX.csv':
            # continue

        # 筛选是否提取交通信号灯（否则为框架和灯都提取）
        if source_annotation_name.split('_') == 'frameAnnotationsBULB.csv':
            continue
        source_annotation_path = os.path.join(
            dataset['source_annotations_folder'], source_annotation_name)

        # 获取data字典中images内的图片信息，file_name、height、width
        total_annotations_dict = multiprocessing.Manager().dict()
        with open(source_annotation_path, 'r') as f:
            pool = multiprocessing.Pool(dataset['workers'])
            for image_base_information in tqdm(csv.reader(f)):
                pool.apply_async(func=lisa.load_image_base_information, args=(
                    dataset, image_base_information, total_annotations_dict,),
                    error_callback=err_call_back)
            pool.close()
            pool.join()
            f.close()

        # 读取目标标注信息
        total_image_segment_list = []
        with open(source_annotation_path, 'r') as f:
            pool = multiprocessing.Pool(dataset['workers'])
            for one_annotation in tqdm(csv.reader(f)):
                total_image_segment_list.append(pool.apply_async(func=lisa.load_image_annotation, args=(
                    dataset, one_annotation, total_annotations_dict,),
                    error_callback=err_call_back))
            pool.close()
            pool.join()
            f.close()

        total_images_data_dict = {}
        for image_true_box in total_image_segment_list:
            if image_true_box.get() is None:
                continue
            if image_true_box.get()[0] not in total_images_data_dict:
                total_images_data_dict[image_true_box.get(
                )[0]] = total_annotations_dict[image_true_box.get()[0]]
                total_images_data_dict[image_true_box.get()[0]].true_box_list.extend(
                    image_true_box.get()[1])
            else:
                total_images_data_dict[image_true_box.get()[0]].true_box_list.extend(
                    image_true_box.get()[1])

        # 输出读取的source annotation至temp annotation
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                        "fail_count": 0,
                                                         "no_true_box_count": 0,
                                                         "temp_file_name_list": process_temp_file_name_list
                                                         })
        pool = multiprocessing.Pool(dataset['workers'])
        for _, image in tqdm(total_images_data_dict.items()):
            pool.apply_async(func=lisa.output_temp_annotation, args=(
                dataset, image, process_output,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 更新输出统计
        success_count += process_output['success_count']
        fail_count += process_output['fail_count']
        no_true_box_count += process_output['no_true_box_count']
        temp_file_name_list += process_output['temp_file_name_list']

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No segmentation delete images: \t {} '.format(no_true_box_count))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n) for n in dataset['class_list_new']))
        f.close()

    return


def SJT_LOAD(dataset: dict) -> None:
    """[数据堂数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                     "fail_count": 0,
                                                     "no_true_box_count": 0,
                                                     "temp_file_name_list": process_temp_file_name_list
                                                     })
    pool = multiprocessing.Pool(dataset['workers'])
    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        pool.apply_async(func=sjt.load_annotation,
                         args=(dataset, source_annotation_name,
                               process_output, sjt.change_Occlusion, sjt.change_traffic_light,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(
        process_output['fail_count']))
    print('No true box delete images: \t {} '.format(
        process_output['no_true_box_count']))
    print('Convert success:           \t {} '.format(
        process_output['success_count']))
    dataset['temp_file_name_list'] = [
        x for x in process_output['temp_file_name_list']]
    # 输出分割类别至temp informations folder
    with open(os.path.join(dataset['temp_informations_folder'], 'segment_classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n)
                              for n in dataset['class_list_new']))
        f.close()

    return


def CVAT_COCO2017_LOAD(dataset) -> None:
    """[COCO2017数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_true_box_count = 0
    temp_file_name_list = []

    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        source_annotation_path = os.path.join(
            dataset['source_annotations_folder'], source_annotation_name)
        with open(source_annotation_path, 'r') as f:
            data = json.loads(f.read())
        class_dict = {}
        for n in data['categories']:
            class_dict['%s' % n['id']] = n['name']

        # 获取data字典中images内的图片信息，file_name、height、width
        total_annotations_dict = multiprocessing.Manager().dict()
        pool = multiprocessing.Pool(dataset['workers'])
        for image_base_information in tqdm(data['images']):
            pool.apply_async(func=cvat_coco2017.load_image_base_information, args=(
                dataset, image_base_information, total_annotations_dict,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 读取目标标注信息
        total_image_box_list = []
        pool = multiprocessing.Pool(dataset['workers'])
        for one_annotation in tqdm(data['annotations']):
            total_image_box_list.append(pool.apply_async(func=cvat_coco2017.load_image_annotation, args=(
                dataset, one_annotation, class_dict, total_annotations_dict,),
                error_callback=err_call_back))
        pool.close()
        pool.join()

        total_images_data_dict = {}
        for image_true_box in total_image_box_list:
            if image_true_box.get() is None:
                continue
            if image_true_box.get()[0] not in total_images_data_dict:
                total_images_data_dict[image_true_box.get(
                )[0]] = total_annotations_dict[image_true_box.get()[0]]
                total_images_data_dict[image_true_box.get()[0]].true_box_list.extend(
                    image_true_box.get()[1])
            else:
                total_images_data_dict[image_true_box.get()[0]].true_box_list.extend(
                    image_true_box.get()[1])

        # 输出读取的source annotation至temp annotation
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                         "fail_count": 0,
                                                         "no_true_box_count": 0,
                                                         "temp_file_name_list": process_temp_file_name_list
                                                         })
        pool = multiprocessing.Pool(dataset['workers'])
        for _, image in tqdm(total_images_data_dict.items()):
            pool.apply_async(func=cvat_coco2017.output_temp_annotation, args=(
                dataset, image, process_output,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 更新输出统计
        success_count += process_output['success_count']
        fail_count += process_output['fail_count']
        no_true_box_count += process_output['no_true_box_count']
        temp_file_name_list += process_output['temp_file_name_list']

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No segmentation delete images: \t {} '.format(no_true_box_count))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n)
                              for n in dataset['class_list_new']))
        f.close()

    return


# def CCPD_LOAD(dataset: dict) -> None:
#     """[CCPD数据集annotation读取]

#     Args:
#         dataset (dict): [数据集信息字典]
#     """

#     src_lab_path = dataset['source_annotations_folder']
#     src_lab_path_list = os.listdir(src_lab_path)  # 读取source label文件夹下的全部文件名
#     success_count = 0
#     fail_count = 0
#     no_true_box_count = 0
#     temp_file_name_list = []
#     print('Start to transform source annotation to temp annotation:')
#     for one_label in tqdm(src_lab_path_list):
#         with open(os.path.join(src_lab_path, one_label), 'r') as f:
#             for n in f.read().splitlines():
#                 true_box_dict_list = []
#                 image_name = n
#                 bbox_info = n.strip('ccpd_').strip('.jpg').split('-')
#                 image_name_new = dataset['file_prefix'] + image_name
#                 image_path = os.path.join(
#                     dataset['temp_images_folder'], image_name_new)
#                 img = cv2.imread(image_path)
#                 if img is None:
#                     print('Can not load: {}'.format(image_name_new))
#                     continue
#                 height, width, channels = img.shape     # 读取每张图片的shape
#                 cls = 'licenseplate'
#                 if cls not in dataset['source_class_list']:
#                     continue
#                 bbox = [int(bbox_info[2].split('_')[0].split('#')[0]),
#                         int(bbox_info[2].split('_')[0].split('#')[1]),
#                         int(bbox_info[2].split('_')[1].split('#')[0]),
#                         int(bbox_info[2].split('_')[1].split('#')[1])]
#                 xmin = min(
#                     max(min(float(bbox[0]), float(bbox[2])), 0.), float(width))
#                 ymin = min(
#                     max(min(float(bbox[1]), float(bbox[3])), 0.), float(height))
#                 xmax = max(
#                     min(max(float(bbox[0]), float(bbox[2])), float(width)), 0.)
#                 ymax = max(
#                     min(max(float(bbox[1]), float(bbox[3])), float(height)), 0.)
#                 true_box_dict_list.append(TRUE_BOX(
#                     cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
#                 image = IMAGE(image_name, image_name_new, image_path, int(
#                     height), int(width), int(channels), true_box_dict_list)
#                 # 输出读取的source annotation至temp annotation
#                 temp_annotation_output_path = os.path.join(
#                     dataset['temp_annotations_folder'],
#                     image.file_name_new + '.' + dataset['temp_annotation_form'])
#                 modify_true_box_list(image, dataset['modify_class_dict'])
#                 if dataset['class_pixel_distance_dict'] is not None:
#                     class_pixel_limit(dataset, image.true_box_list)
#                 if 0 == len(image.true_box_list):
#                     print('{} no true box, has been delete.'.format(
#                         image.image_name_new))
#                     os.remove(image.image_path)
#                     no_true_box_count += 1
#                     continue
#                 if TEMP_OUTPUT(temp_annotation_output_path, image):
#                     temp_file_name_list.append(image.file_name_new)
#                     success_count += 1
#                 else:
#                     no_true_box_count += 1

#     print('\nSource dataset convert to temp dataset file count: ')
#     print('Total annotations:         \t {} '.format(
#         len(os.listdir(dataset['source_annotations_folder']))))
#     print('Convert fail:              \t {} '.format(fail_count))
#     print('No true box delete images: \t {} '.format(no_true_box_count))
#     print('Convert success:           \t {} '.format(success_count))
#     dataset['temp_file_name_list'] = temp_file_name_list
#     with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
#         if len(dataset['class_list_new']):
#             f.write('\n'.join(str(n) for n in dataset['class_list_new']))
#         f.close()

#     return


# def LICENSEPLATE_LOAD(dataset: dict):
#     """抽取源标签文件中真实框信息声明为class per_image，返回total_images_data_list"""

#     local_mask = {"皖": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9,
#                   "苏": 10, "浙": 11, "京": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18,
#                   "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24, "西": 25, "陕": 26, "甘": 27,
#                   "青": 28, "宁": 29, "新": 30}

#     code_mask = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "j": 8, "k": 9,
#                  "l": 10, "m": 11, "n": 12, "p": 13, "q": 14, "r": 15, "s": 16, "t": 17, "u": 18,
#                  "v": 19, "w": 20, "x":  21, "y": 22, "z": 23, "0_": 24, "1": 25, "2": 26, "3": 27,
#                  "4": 28, "5": 29, "6": 30, "7": 31, "8": 32, "9": 33}

#     local_mask_key_list = [x for x, _ in local_mask.items()]
#     code_mask_key_list = [x for x, _ in code_mask.items()]

#     success_count = 0
#     fail_count = 0
#     no_true_box_count = 0
#     temp_file_name_list = []
#     print('Start to transform source annotation to temp annotation:')
#     # 将每一个source label文件转换为per_image类
#     for src_lab_path_one in tqdm(os.listdir(
#             dataset['source_annotations_folder'])):
#         temp_annotation_output_path = os.path.join(
#             dataset['temp_annotations_folder'],
#             dataset['file_prefix'] + src_lab_path_one)
#         src_lab_dir = os.path.join(
#             dataset['source_annotations_folder'], src_lab_path_one)
#         with open(src_lab_dir, 'r') as f:
#             truebox_dict_list = []
#             for one_bbox in f.read().splitlines():
#                 bbox = one_bbox.split(' ')[1:]
#                 image_name = (src_lab_dir.split(
#                     '/')[-1]).replace('.txt', '.jpg')
#                 image_name_new = dataset['file_prefix'] + (src_lab_dir.split(
#                     '/')[-1]).replace('.txt', '.jpg')
#                 image_path = os.path.join(
#                     dataset['temp_images_folder'], image_name_new)
#                 img = cv2.imread(image_path)
#                 if img is None:
#                     print('Can not load: {}'.format(image_name_new))
#                     continue
#                 size = img.shape
#                 width = int(size[1])
#                 height = int(size[0])
#                 channels = int(size[2])
#                 cls = dataset['source_class_list'][int(one_bbox.split(' ')[0])]
#                 cls = cls.strip(' ').lower()
#                 if cls not in dataset['source_class_list']:
#                     continue
#                 if cls == 'dontcare' or cls == 'misc':
#                     continue
#                 bbox = revers_yolo(size, bbox)
#                 xmin = min(
#                     max(min(float(bbox[0]), float(bbox[1])), 0.), float(width))
#                 ymin = min(
#                     max(min(float(bbox[2]), float(bbox[3])), 0.), float(height))
#                 xmax = max(
#                     min(max(float(bbox[1]), float(bbox[0])), float(width)), 0.)
#                 ymax = max(
#                     min(max(float(bbox[3]), float(bbox[2])), float(height)), 0.)
#                 truebox_dict_list.append(TRUE_BOX(
#                     cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
#         if 7 != len(truebox_dict_list):
#             continue
#         truebox_dict_list.sort(key=lambda x: x.xmin)

#         # 更换真实框类别为车牌真实值
#         real_classes_list = list(
#             map(int, src_lab_path_one.split('-')[4].split('_')))
#         classes_decode_list = []
#         classes_decode_list.append(local_mask_key_list[real_classes_list[0]])
#         for one in real_classes_list[1:]:
#             classes_decode_list.append(code_mask_key_list[one])
#         for truebox, classes in zip(truebox_dict_list, classes_decode_list):
#             truebox.clss = classes
#         image = IMAGE(image_name, image_name_new, image_path, int(
#             height), int(width), int(channels), truebox_dict_list)
#         # 将单张图对象添加进全数据集数据列表中
#         modify_true_box_list(image, dataset['modify_class_dict'])
#         if dataset['class_pixel_distance_dict'] is not None:
#             class_pixel_limit(dataset, image.true_box_list)
#         if 0 == len(image.true_box_list):
#             print('{} no true box, has been delete.'.format(image.image_name_new))
#             os.remove(image_path)
#             no_true_box_count += 1
#             continue
#         # 输出读取的source annotation至temp annotation
#         if TEMP_OUTPUT(temp_annotation_output_path, image):
#             temp_file_name_list.append(image.file_name_new)
#             success_count += 1
#         else:
#             no_true_box_count += 1
#     print('\nSource dataset convert to temp dataset file count: ')
#     print('Total annotations:         \t {} '.format(
#         len(os.listdir(dataset['source_annotations_folder']))))
#     print('Convert fail:              \t {} '.format(fail_count))
#     print('No true box delete images: \t {} '.format(no_true_box_count))
#     print('Convert success:           \t {} '.format(success_count))
#     dataset['temp_file_name_list'] = temp_file_name_list
#     with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
#         if len(dataset['class_list_new']):
#             f.write('\n'.join(str(n) for n in dataset['class_list_new']))
#         f.close()

#     return


# def random_test(pre_data_list):
#     """[为无距离、遮挡属性的数据添加随机距离和遮挡率]
#     Args:
#         pre_data_list ([list]): [读取的图片类信息列表]
#     Returns:
#         [list]: [随机修改距离和遮挡率的图片类信息列表]
#     """
#     distance_list = [0, 50, 100]
#     occlusion_list = [0, 35, 50, 75]
#     data_list = pre_data_list.copy()
#     for one_image_info in data_list:
#         for one_true_box in one_image_info.true_box_list:
#             one_true_box.distance = random.sample(distance_list, 1)[0]
#             one_true_box.occlusion = random.sample(occlusion_list, 1)[0]
#     return data_list


annotation_load_function_dict = {'pascal_voc': PASCAL_VOC_LOAD,
                                 'coco2017': COCO2017_LOAD,
                                 'tt100k': TT100K_LOAD,
                                 'kitti': KITTI_LOAD,
                                 'cctsdb': CCTSDB_LOAD,
                                 'lisa': LISA_LOAD,
                                 'sjt': SJT_LOAD,
                                 'hy': HY_DATASET_LOAD,
                                 'myxb': MYXB_LOAD,
                                 'hy_highway': HY_HIGHWAY_LOAD,
                                 'yolo': YOLO_LOAD,
                                 'cvat_coco2017': CVAT_COCO2017_LOAD,
                                 'huawei_segment': HUAWEI_SEGMENT_LOAD,
                                 #  'ccpd': CCPD_LOAD,
                                 #  'licenseplate': LICENSEPLATE_LOAD
                                 }


def annotation_load_function(dataset_stype, *args):
    """[根据输入类别挑选数据集提取、转换函数]

    Returns:
        [function]: [读取数据集标签函数]
    """

    return annotation_load_function_dict.get(dataset_stype)(*args)
