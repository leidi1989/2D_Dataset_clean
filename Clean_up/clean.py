'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:45:50
LastEditors: Leidi
LastEditTime: 2022-01-10 18:53:00
'''
import os
import time
import yaml
import argparse
import multiprocessing

import dataset


def main(dataset_info: dict) -> None:
    """[数据集清理]

    Args:
        dataset_info (dict): [数据集信息字典]
    """

    return


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='clean.py')
    parser.add_argument('--config', '--c', dest='config', default=r'/home/leidi/hy_program/2D_Dataset_clean/Clean_up/config/default.yaml',
                        type=str, help='dataset config file path')
    parser.add_argument('--workers', '--w', dest='workers', default=multiprocessing.cpu_count(),
                        type=int, help='maximum number of dataloader workers(multiprocessing.cpu_count())')

    opt = parser.parse_args()
    # load dataset config file
    dataset_config = yaml.load(
        open(opt.config, 'r', encoding="utf-8"), Loader=yaml.FullLoader)
    dataset_config.update({'workers': opt.workers})
    
    Input_dataset = dataset.__dict__[dataset_config['Source_dataset_style']](dataset_config)
    Input_dataset.source_dataset_copy_image_and_annotation()
    Input_dataset.transform_to_temp_dataset()
    pass