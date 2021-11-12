'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-05 21:50:49
LastEditors: Leidi
LastEditTime: 2021-10-20 17:30:29
'''
from tqdm import tqdm
import multiprocessing

from utils.utils import *
from annotation.dataset_output_function import bdd100k, yolop


def BDD100K_OUTPUT(dataset: dict) -> None:
    """[输出temp dataset annotation为BDD100K]

     Args:
         dataset (dict): [temp dataset]
    """

    print('Start output temp dataset annotations to YOLOP annotations:')
    pool = multiprocessing.Pool(dataset['workers'])
    for temp_annotation_path in tqdm(dataset['temp_annotation_path_list']):
        pool.apply_async(func=bdd100k.annotation_output,
                         args=(dataset, temp_annotation_path,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    return


def YOLOP_OUTPUT(dataset: dict) -> None:
    """[输出temp dataset annotation为YOLOP标注]

     Args:
         dataset (dict): [数据集信息字典]
    """

    print('Start output temp dataset annotations to YOLOP annotations:')
    pool = multiprocessing.Pool(dataset['workers'])
    for temp_annotation_path in tqdm(dataset['temp_annotation_path_list']):
        pool.apply_async(func=yolop.annotation_output,
                         args=(dataset, temp_annotation_path,),
                         error_callback=err_call_back)
    pool.close()
    pool.join()

    return


annotation_out_function_dict = {'bdd100k': BDD100K_OUTPUT,
                                'yolop': YOLOP_OUTPUT,
                                }


def annotation_output_funciton(dataset_stype: str, *args):
    """[获取指定类别数据集annotation输出函数。]

    Args:
        dataset_style (str): [输出数据集类别。]

    Returns:
        [function]: [返回指定类别数据集输出函数。]
    """

    return annotation_out_function_dict.get(dataset_stype)(*args)
