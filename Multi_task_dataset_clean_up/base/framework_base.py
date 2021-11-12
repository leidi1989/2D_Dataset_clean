'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-10 20:06:35
LastEditors: Leidi
LastEditTime: 2021-09-19 19:52:24
'''
from out.framework_update import framework_funciton


def framework(dataset: dict) -> None:
    """[调整暂存数据集为对于目标数据集组织结构]

    Args:
        dataset (dict): [数据集信息字典]
    """    
    framework_funciton(
        dataset['target_dataset_style'], dataset)

    return