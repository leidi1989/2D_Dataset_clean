'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:45:50
LastEditors: Leidi
LastEditTime: 2022-02-06 00:46:41
'''
import time
import yaml
import argparse
import multiprocessing

import dataset


def main(dataset_config: dict) -> None:
    """[数据集清理]

    Args:
        dataset_info (dict): [数据集信息字典]
    """
    # try:
    Input_dataset = dataset.__dict__[
        dataset_config['Source_dataset_style']](dataset_config)
    # except:
    #     print('Dataset initialize wrong, abort.')
        
    Input_dataset.source_dataset_copy_image_and_annotation()
    Input_dataset.transform_to_temp_dataset()
    Input_dataset.output_classname_file()
    Input_dataset.delete_redundant_image()
    Input_dataset.get_dataset_information()

    dataset.__dict__[dataset_config['Target_dataset_style']
                     ].target_dataset(Input_dataset)

    Input_dataset.target_dataset_annotation_check()

    dataset.__dict__[dataset_config['Target_dataset_style']
                     ].target_dataset_folder(Input_dataset)

    return


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(prog='clean.py')
    parser.add_argument('--config', '--c', dest='config', default=r'Clean_up/config/default.yaml',
                        type=str, help='dataset config file path')
    parser.add_argument('--workers', '--w', dest='workers', default=multiprocessing.cpu_count(),
                        type=int, help='maximum number of dataloader workers(multiprocessing.cpu_count())')

    opt = parser.parse_args()
    # load dataset config file
    dataset_config = yaml.load(
        open(opt.config, 'r', encoding="utf-8"), Loader=yaml.FullLoader)
    dataset_config.update({'workers': opt.workers})

    main(dataset_config)
