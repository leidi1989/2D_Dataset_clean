'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:43:21
LastEditors: Leidi
LastEditTime: 2021-12-13 16:52:32
'''
import os
import json
import multiprocessing

from utils.utils import *
import annotation.dataset_load_function as F


def cvat_image_1_1(dataset: dict) -> None:
    """[cvat_image_1_1数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                     "fail_count": 0,
                                                     "no_segmentation": 0,
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
        process_output['no_segmentation']))
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


def cityscapes(dataset: dict) -> None:
    """[Cityscapes数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                     "fail_count": 0,
                                                     "no_segmentation": 0,
                                                     "temp_file_name_list": process_temp_file_name_list
                                                     })
    pool = multiprocessing.Pool(dataset['workers'])
    for source_annotations_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
        pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_annotation, args=(
            dataset, source_annotations_name, process_output,),
            error_callback=err_call_back)
    pool.close()
    pool.join()

    print('\nSource dataset convert to temp dataset file count: ')
    print('Total annotations:         \t {} '.format(
        len(os.listdir(dataset['source_annotations_folder']))))
    print('Convert fail:              \t {} '.format(
        process_output['fail_count']))
    print('No segmentation delete images: \t {} '.format(
        process_output['no_segmentation']))
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


def bdd100k(dataset: dict) -> None:
    """[BDD100K割数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                     "fail_count": 0,
                                                     "no_segmentation": 0,
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
        process_output['no_segmentation']))
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


def coco2017(dataset: dict) -> None:
    """[COCO2017数据集annotation读取]

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
        
        del f

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
        total_image_segment_list = []
        pool = multiprocessing.Pool(dataset['workers'])
        for one_annotation in tqdm(data['annotations']):
            total_image_segment_list.append(pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_image_annotation, args=(
                dataset, one_annotation, class_dict, total_annotations_dict,),
                error_callback=err_call_back))
        pool.close()
        pool.join()
        
        del data

        total_images_data_dict = {}
        for image_true_segment in total_image_segment_list:
            if image_true_segment.get() is None:
                continue
            if image_true_segment.get()[0] not in total_images_data_dict:
                total_images_data_dict[image_true_segment.get(
                )[0]] = total_annotations_dict[image_true_segment.get()[0]]
                total_images_data_dict[image_true_segment.get()[0]].true_segmentation_list.extend(
                    image_true_segment.get()[1])
            else:
                total_images_data_dict[image_true_segment.get()[0]].true_segmentation_list.extend(
                    image_true_segment.get()[1])
        
        del total_annotations_dict, total_image_segment_list

        # 输出读取的source annotation至temp annotation
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                         "fail_count": 0,
                                                         "no_segmentation": 0,
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
            f.write('\n'.join(str(n)
                              for n in dataset['class_list_new']))
        f.close()

    return


def apolloscape_lane_segment(dataset: dict) -> None:
    """[APOLLOSCAPE_LANE_SEGMENT数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                     "fail_count": 0,
                                                     "no_segmentation": 0,
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
        process_output['no_segmentation']))
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


def huaweiyun_segment(dataset: dict) -> None:
    """[HUAWEI SEGMENT数据集annotation读取]

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
            
        del f
        
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
        pool = multiprocessing.Pool(dataset['workers'])
        total_image_segment_list = []
        for one_annotation in tqdm(data['annotations']):
            total_image_segment_list.append(pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_image_annotation, args=(
                dataset, one_annotation, class_dict, total_annotations_dict,),
                error_callback=err_call_back))
        pool.close()
        pool.join()
        
        del data

        total_images_data_dict = {}
        for image_segment in total_image_segment_list:
            if image_segment.get() is None:
                continue
            if image_segment.get()[0] not in total_images_data_dict:
                total_images_data_dict[image_segment.get(
                )[0]] = total_annotations_dict[image_segment.get()[0]]
                total_images_data_dict[image_segment.get()[0]].true_segmentation_list.extend(
                    image_segment.get()[1])
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
        for _, image in tqdm(total_images_data_dict.items()):
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


# def yunce_segment(dataset: dict) -> None:
#     """[HUAWEI SEGMENT数据集annotation读取]

#     Args:
#         dataset (dict): [数据集信息字典]
#     """

#     success_count = 0
#     fail_count = 0
#     no_segmentation = 0
#     temp_file_name_list = []

#     for source_annotation_name in tqdm(os.listdir(dataset['source_annotations_folder'])):
#         source_annotation_path = os.path.join(
#             dataset['source_annotations_folder'], source_annotation_name)
#         with open(source_annotation_path, 'r') as f:
#             data = json.loads(f.read())
#             f.close()

#         del f
            
#         class_dict = {}
#         for n in data['categories']:
#             class_dict['%s' % n['id']] = n['name']
            
#         annotation_dict = {}
#         for n in tqdm(data['annotations']):
#             source_image_name = os.path.splitext(n['image_name'])[0]
#             if source_image_name not in annotation_dict:
#                 annotation_dict.update({source_image_name:[n]})
#             else:
#                 annotation_dict[source_image_name].append(n)

#         # 获取data字典中images内的图片信息，file_name、height、width
#         process_temp_file_name_list = multiprocessing.Manager().list()
#         process_output = multiprocessing.Manager().dict({"success_count": 0,
#                                                          "fail_count": 0,
#                                                          "no_segmentation": 0,
#                                                          "temp_file_name_list": process_temp_file_name_list
#                                                          })
#         pool = multiprocessing.Pool(dataset['workers'])
#         for source_image_name, image_annotation in tqdm(annotation_dict.items()):
#             pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_image_annotation, args=(
#                 dataset, source_image_name, image_annotation, process_output,),
#                 error_callback=err_call_back)
#         pool.close()
#         pool.join()
        
#         # 更新输出统计
#         success_count += process_output['success_count']
#         fail_count += process_output['fail_count']
#         no_segmentation += process_output['no_segmentation']
#         temp_file_name_list += process_output['temp_file_name_list']
        
#     # 输出读取统计结果
#     print('\nSource dataset convert to temp dataset file count: ')
#     print('Total annotations:         \t {} '.format(
#         len(os.listdir(dataset['source_annotations_folder']))))
#     print('Convert fail:              \t {} '.format(fail_count))
#     print('No segmentation delete images: \t {} '.format(no_segmentation))
#     print('Convert success:           \t {} '.format(success_count))
#     dataset['temp_file_name_list'] = temp_file_name_list
#     with open(os.path.join(dataset['temp_informations_folder'], 'classes.names'), 'w') as f:
#         if len(dataset['class_list_new']):
#             f.write('\n'.join(str(n) for n in dataset['class_list_new']))
#         f.close()
        
#     return


def yunce_segment(dataset: dict) -> None:
    """[HUAWEI SEGMENT数据集annotation读取]

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
        total_image_segment_list = []
        for one_annotation in tqdm(data['annotations']):
            total_image_segment_list.append(pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].load_image_annotation, args=(
                dataset, one_annotation, class_dict, total_annotations_dict,),
                error_callback=err_call_back))
        pool.close()
        pool.join()
        
        del data
        
        total_images_data_dict = {}
        for image_segment in total_image_segment_list:
            if image_segment.get() is None:
                continue
            if image_segment.get()[0] not in total_images_data_dict:
                try:
                    total_images_data_dict.update({image_segment.get(
                    )[0]: total_annotations_dict[image_segment.get()[0]]})
                    total_images_data_dict[image_segment.get()[0]].true_segmentation_list.extend(
                        image_segment.get()[1])
                except:
                    print('\nGet temp information erro:', image_segment.get()[0])
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


def hy_val(dataset: dict) -> None:
    """[Cityscapes数据集annotation读取]

    Args:
        dataset (dict): [数据集信息字典]
    """

    process_temp_file_name_list = multiprocessing.Manager().list()
    process_output = multiprocessing.Manager().dict({"success_count": 0,
                                                     "fail_count": 0,
                                                     "no_segmentation": 0,
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
        process_output['no_segmentation']))
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


# annotation_load_function_dict = {'coco2017': COCO_2017_LOAD,
#                                  'cityscapes': CITYSCAPES_LOAD,
#                                  'bdd100k': BDD100K_LOAD,
#                                  'apolloscape_lane_segment': APOLLOSCAPE_LANE_SEGMENT_LOAD,
#                                  'huawei_segment': HUAWEI_SEGMENT_LOAD,
#                                  'hy_val': CITYSCAPESVAL_LOAD
#                                  }


# def annotation_load_function(dataset_stype: str, *args):
#     """[获取指定类别数据集annotation提取函数。]

#     Args:
#         dataset_style (str): [输出数据集类别。]

#     Returns:
#         [function]: [返回指定类别数据集读取函数。]
#     """

#     return annotation_load_function_dict.get(dataset_stype)(*args)
