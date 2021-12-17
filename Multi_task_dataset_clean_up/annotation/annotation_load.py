'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:43:21
LastEditors: Leidi
LastEditTime: 2021-12-17 18:22:55
'''
import os
import json
import multiprocessing

from utils.utils import *
import annotation.dataset_load_function as F


def huawei_segment(dataset: dict) -> None:
    """[华为标注分割数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_detect_segmentation = 0
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
        print('Load image base informations:')
        for image_base_information in tqdm(data['images']):
            pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_image_base_information, args=(
                dataset, image_base_information, total_annotations_dict,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 读取目标标注信息
        total_image_box_segment_list = []
        pool = multiprocessing.Pool(dataset['workers'])
        print('Load image true and segment annotation:')
        for one_annotation in tqdm(data['annotations']):
            total_image_box_segment_list.append(pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_image_annotation, args=(
                dataset, one_annotation, class_dict, total_annotations_dict,),
                error_callback=err_call_back))
        pool.close()
        pool.join()

        total_images_data_dict = {}
        for image_segment in total_image_box_segment_list:
            if image_segment.get() is None:
                continue
            if image_segment.get()[0] not in total_images_data_dict:
                total_images_data_dict[image_segment.get(
                )[0]] = total_annotations_dict[image_segment.get()[0]]
                total_images_data_dict[image_segment.get()[0]].true_box_list.extend(
                    image_segment.get()[1])
                total_images_data_dict[image_segment.get()[0]].true_segmentation_list.extend(
                    image_segment.get()[2])
            else:
                total_images_data_dict[image_segment.get()[0]].true_box_list.extend(
                    image_segment.get()[1])
                total_images_data_dict[image_segment.get()[0]].true_segmentation_list.extend(
                    image_segment.get()[2])

        # 输出读取的source annotation至temp annotation
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                         "fail_count": 0,
                                                         "no_detect_segmentation": 0,
                                                         "temp_file_name_list": process_temp_file_name_list
                                                         })
        pool = multiprocessing.Pool(dataset['workers'])
        for _, image in tqdm(total_images_data_dict.items()):
            pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].output_temp_annotation, args=(
                dataset, image, process_output,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 更新输出统计
        success_count += process_output['success_count']
        fail_count += process_output['fail_count']
        no_detect_segmentation += process_output['no_detect_segmentation']
        temp_file_name_list += process_output['temp_file_name_list']

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No segmentation delete images: \t {} '.format(no_detect_segmentation))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'detect_classes.names'), 'w') as f:
        if len(dataset['detect_class_list_new']):
            f.write('\n'.join(str(n)
                    for n in dataset['detect_class_list_new']))
        f.close()

    with open(os.path.join(dataset['temp_informations_folder'], 'segment_classes.names'), 'w') as f:
        if len(dataset['segment_class_list_new']):
            f.write('\n'.join(str(n)
                    for n in dataset['segment_class_list_new']))
        f.close()

    return


def coco2017(dataset: dict) -> None:
    """[COCO2017数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_detect_segmentation = 0
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
            pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_image_base_information, args=(
                dataset, image_base_information, total_annotations_dict,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 读取目标标注信息
        total_image_box_segment_list = []
        pool = multiprocessing.Pool(dataset['workers'])
        for one_annotation in tqdm(data['annotations']):
            total_image_box_segment_list.append(pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_image_annotation, args=(
                dataset, one_annotation, class_dict, total_annotations_dict,),
                error_callback=err_call_back))
        pool.close()
        pool.join()

        total_images_data_dict = {}
        for image_true_box_segment in total_image_box_segment_list:
            if image_true_box_segment.get() is None:
                continue
            if image_true_box_segment.get()[0] not in total_images_data_dict:
                total_images_data_dict[image_true_box_segment.get(
                )[0]] = total_annotations_dict[image_true_box_segment.get()[0]]
                total_images_data_dict[image_true_box_segment.get()[0]].true_box_list.extend(
                    image_true_box_segment.get()[1])
                total_images_data_dict[image_true_box_segment.get()[0]].true_segmentation_list.extend(
                    image_true_box_segment.get()[2])
            else:
                total_images_data_dict[image_true_box_segment.get()[0]].true_box_list.extend(
                    image_true_box_segment.get()[1])
                total_images_data_dict[image_true_box_segment.get()[0]].true_segmentation_list.extend(
                    image_true_box_segment.get()[2])

        # 输出读取的source annotation至temp annotation
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                         "fail_count": 0,
                                                         "no_detect_segmentation": 0,
                                                         "temp_file_name_list": process_temp_file_name_list
                                                         })
        pool = multiprocessing.Pool(dataset['workers'])
        for _, image in tqdm(total_images_data_dict.items()):
            pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].output_temp_annotation, args=(
                dataset, image, process_output,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 更新输出统计
        success_count += process_output['success_count']
        fail_count += process_output['fail_count']
        no_detect_segmentation += process_output['no_detect_segmentation']
        temp_file_name_list += process_output['temp_file_name_list']

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No segmentation delete images: \t {} '.format(no_detect_segmentation))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'detect_classes.names'), 'w') as f:
        if len(dataset['detect_class_list_new']):
            f.write('\n'.join(str(n)
                    for n in dataset['detect_class_list_new']))
        f.close()

    with open(os.path.join(dataset['temp_informations_folder'], 'segment_classes.names'), 'w') as f:
        if len(dataset['segment_class_list_new']):
            f.write('\n'.join(str(n)
                    for n in dataset['segment_class_list_new']))
        f.close()

    return


def bdd100k(dataset: dict) -> None:
    """[BDD100K割数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    process_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                     "fail_count": 0,
                                                     "no_detect_segmentation": 0,
                                                     "temp_file_name_list": process_list
                                                     })
    pool = multiprocessing.Pool(dataset['workers'])
    for source_annotation_path in tqdm(os.listdir(dataset['source_annotations_folder'])):
        pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_annotation, args=(
            dataset, source_annotation_path, process_output,),
            error_callback=err_call_back)
    pool.close()
    pool.join()

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(
        process_output['fail_count']))
    print('No segmentation delete images: \t {} '.format(
        process_output['no_detect_segmentation']))
    print('Convert success:           \t {} '.format(
        process_output['success_count']))
    dataset['temp_file_name_list'] = [
        x for x in process_output['temp_file_name_list']]
    # 输出真实框、分割类别至temp informations folder
    with open(os.path.join(dataset['temp_informations_folder'], 'detect_classes.names'), 'w') as f:
        if len(dataset['detect_class_list_new']):
            f.write('\n'.join(str(n)
                              for n in dataset['detect_class_list_new']))
        f.close()
    with open(os.path.join(dataset['temp_informations_folder'], 'segment_classes.names'), 'w') as f:
        if len(dataset['segment_class_list_new']):
            f.write('\n'.join(str(n)
                              for n in dataset['segment_class_list_new']))
        f.close()

    return


def yunce_segment_coco(dataset: dict) -> None:
    """[yunce_segment_coco数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    success_count = 0
    fail_count = 0
    no_segmentation = 0
    temp_file_name_list = []

    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        source_annotation_path = os.path.join(
            dataset['source_annotations_folder'], source_annotation_name)
        with open(source_annotation_path, 'r') as f:
            data = json.loads(f.read())
            f.close()

        del f

        class_dict = {}
        for n in data['categories']:
            class_dict['%s' % n['id']] = n['name']

        # 获取data字典中images内的图片信息，file_name、height、width
        total_annotations_dict = multiprocessing.Manager().dict()
        pool = multiprocessing.Pool(dataset['workers'])
        for image_base_information in data['images']:
            pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_image_base_information, args=(
                dataset, image_base_information, total_annotations_dict,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 读取目标标注信息
        pool = multiprocessing.Pool(dataset['workers'])
        total_image_detect_segment_list = []
        for one_annotation in data['annotations']:
            total_image_detect_segment_list.append(pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_image_annotation, args=(
                dataset, one_annotation, class_dict, total_annotations_dict,),
                error_callback=err_call_back))
        pool.close()
        pool.join()

        del data

        total_images_data_dict = {}
        for image_detect_segment in total_image_detect_segment_list:
            if image_segment.get() is None:
                continue
            if image_segment.get()[0] not in total_images_data_dict:
                try:
                    total_images_data_dict.update({image_segment.get(
                    )[0]: total_annotations_dict[image_segment.get()[0]]})
                    total_images_data_dict[image_segment.get()[0]].true_segmentation_list.extend(
                        image_segment.get()[1])
                except:
                    print('\nGet temp information erro:',
                          image_segment.get()[0])
                    continue
            else:
                total_images_data_dict[image_segment.get()[0]].true_segmentation_list.extend(
                    image_segment.get()[1])

        del total_annotations_dict, total_image_segment_list

        # 输出读取的source annotation至temp annotation
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                         "fail_count": 0,
                                                         "no_segmentation": 0,
                                                         "temp_file_name_list": process_temp_file_name_list
                                                         })
        pool = multiprocessing.Pool(dataset['workers'])
        for _, image in total_images_data_dict.items():
            pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].output_temp_annotation, args=(
                dataset, image, process_output,),
                error_callback=err_call_back)
        pool.close()
        pool.join()

        # 更新输出统计
        success_count += process_output['success_count']
        fail_count += process_output['fail_count']
        no_segmentation += process_output['no_segmentation']
        temp_file_name_list += process_output['temp_file_name_list']

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(fail_count))
    print('No segmentation delete images: \t {} '.format(no_segmentation))
    print('Convert success:           \t {} '.format(success_count))
    dataset['temp_file_name_list'] = temp_file_name_list
    with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
        if len(dataset['class_list_new']):
            f.write('\n'.join(str(n) for n in dataset['class_list_new']))
        f.close()

    return


def yunce_segment_coco_one_image(dataset: dict) -> None:
    """[yunce_segment_coco_one_image数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                     "fail_count": 0,
                                                     "no_detect_segmentation": 0,
                                                     "temp_file_name_list": process_temp_file_name_list
                                                     })
    pool = multiprocessing.Pool(dataset['workers'])
    for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_annotation,
                         args=(dataset, source_annotation_name,
                               process_output,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    # 输出读取统计结果
    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(
        process_output['fail_count']))
    print('No segmentation delete images: \t {} '.format(
        process_output['no_detect_segmentation']))
    print('Convert success:           \t {} '.format(
        process_output['success_count']))
    dataset['temp_file_name_list'] = [
        x for x in process_output['temp_file_name_list']]
    # 输出真实框、分割类别至temp informations folder
    with open(os.path.join(dataset['temp_informations_folder'], 'detect_classes.names'), 'w') as f:
        if len(dataset['detect_class_list_new']):
            f.write('\n'.join(str(n)
                              for n in dataset['detect_class_list_new']))
        f.close()
    with open(os.path.join(dataset['temp_informations_folder'], 'segment_classes.names'), 'w') as f:
        if len(dataset['segment_class_list_new']):
            f.write('\n'.join(str(n)
                              for n in dataset['segment_class_list_new']))
        f.close()

    return
