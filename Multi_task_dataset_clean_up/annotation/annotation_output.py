'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-05 21:50:49
LastEditors: Leidi
LastEditTime: 2021-12-24 17:07:56
'''
import time
from tqdm import tqdm
import multiprocessing

from utils.utils import *
import annotation.dataset_output_function as F


def bdd100k(dataset: dict) -> None:
    """[输出temp dataset annotation为BDD100K]

     Args:
         dataset (dict): [temp dataset]
    """

    print('Start output temp dataset annotations to YOLOP annotations:')
    pool = multiprocessing.Pool(dataset['workers'])
    for temp_annotation_path in tqdm(dataset['temp_annotation_path_list']):
        pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].annotation_output,
                         args=(dataset, temp_annotation_path,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    return


def yolop(dataset: dict) -> None:
    """[输出temp dataset annotation为YOLOP标注]

     Args:
         dataset (dict): [数据集信息字典]
    """

    print('Start output temp dataset annotations to YOLOP annotations:')
    pool = multiprocessing.Pool(dataset['workers'])
    for temp_annotation_path in tqdm(dataset['temp_annotation_path_list']):
        pool.apply_async(func=F.__dict__[dataset['source_dataset_stype']].annotation_output,
                         args=(dataset, temp_annotation_path,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    return


def coco2017(dataset) -> None:
    """[输出temp dataset annotation]

    Args:
        dataset (Dataset): [temp dataset]
    """

    print('Start output target annotations:')
    for dataset_temp_annotation_path_list in tqdm(dataset['temp_divide_file_list'][1:-1]):
        # 声明coco字典及基础信息
        coco = {'info': {'description': 'COCO 2017 Dataset',
                         'url': 'http://cocodataset.org',
                         'version': '1.0',
                         'year': 2017,
                         'contributor': 'leidi',
                         'date_created': time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
                         },
                'licenses': [
            {
                'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
                'id': 1,
                'name': 'Attribution-NonCommercial-ShareAlike License'
            },
            {
                'url': 'http://creativecommons.org/licenses/by-nc/2.0/',
                'id': 2,
                'name': 'Attribution-NonCommercial License'
            },
            {
                'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/',
                'id': 3,
                'name': 'Attribution-NonCommercial-NoDerivs License'
            },
            {
                'url': 'http://creativecommons.org/licenses/by/2.0/',
                'id': 4,
                'name': 'Attribution License'
            },
            {
                'url': 'http://creativecommons.org/licenses/by-sa/2.0/',
                'id': 5,
                'name': 'Attribution-ShareAlike License'
            },
            {
                'url': 'http://creativecommons.org/licenses/by-nd/2.0/',
                'id': 6,
                'name': 'Attribution-NoDerivs License'
            },
            {
                'url': 'http://flickr.com/commons/usage/',
                'id': 7,
                'name': 'No known copyright restrictions'
            },
            {
                'url': 'http://www.usa.gov/copyright.shtml',
                'id': 8,
                'name': 'United States Government Work'
            }
        ],
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # 将class_list_new转换为coco格式字典
        for n, cls in enumerate(dataset['detect_class_list_new'] + dataset['segment_class_list_new']):
            category_item = {'supercategory': 'none',
                             'id': n,
                             'name': cls}
            coco['categories'].append(category_item)

        annotation_output_path = os.path.join(
            dataset['target_annotations_folder'], os.path.splitext(
                dataset_temp_annotation_path_list.split(os.sep)[-1])[0]
            + str(2017) + '.' + dataset['target_annotation_form'])
        annotation_path_list = []
        with open(dataset_temp_annotation_path_list, 'r') as f:
            for n in f.readlines():
                annotation_path_list.append(n.replace('\n', '')
                                            .replace(dataset['source_images_folder'], dataset['temp_annotations_folder'])
                                            .replace(dataset['target_image_form'], dataset['temp_annotation_form']))

        # 读取标签图片基础信息
        print('Start load image information:')
        image_information_list = []
        pool = multiprocessing.Pool(dataset['workers'])
        for n, temp_annotation_path in tqdm(enumerate(annotation_path_list)):
            image_information_list.append(pool.apply_async(func=F.__dict__[dataset['target_dataset_style']].get_image_information,
                                                           args=(
                                                               dataset, coco, n, temp_annotation_path,),
                                                           error_callback=err_call_back))
        pool.close()
        pool.join()

        for n in tqdm(image_information_list):
            coco['images'].append(n.get())
        del image_information_list
        
        # 读取标签标注基础信息
        print('Start load annotation:')
        annotations_list = []
        pool = multiprocessing.Pool(dataset['workers'])
        process_annotation_count = multiprocessing.Manager().dict(
            {'annotation_count': 0})
        for n, temp_annotation_path in tqdm(enumerate(annotation_path_list)):
            annotations_list.append(pool.apply_async(func=F.__dict__[dataset['target_dataset_style']].get_annotation,
                                                     args=(dataset, n,
                                                           temp_annotation_path, process_annotation_count,),
                                                     error_callback=err_call_back))
        pool.close()
        pool.join()
        
        annotation_id = 0
        for n in tqdm(annotations_list):
            for m in n.get():
                m['id'] = annotation_id
                coco['annotations'].append(m)
                annotation_id += 1
        del annotations_list

        json.dump(coco, open(annotation_output_path, 'w'))

    return
