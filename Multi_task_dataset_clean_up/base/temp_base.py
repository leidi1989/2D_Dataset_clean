'''
Description:
Version:
Author: Leidi
Date: 2021-08-03 22:18:39
LastEditors: Leidi
LastEditTime: 2021-10-28 11:17:26
'''
from annotation.annotation_load import annotation_load_function


def temp(dataset: dict) -> None:
    """[创建目标数据集temp_annotation]

    Args:
        dataset (dict): [数据集信息字典]
    """
    
    print('\nStart to transform source annotation to temp annotation:')
    annotation_load_function(
        dataset['source_dataset_stype'], dataset)
    
    return
