'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-05 21:50:49
LastEditors: Leidi
LastEditTime: 2021-11-15 15:21:52
'''
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
