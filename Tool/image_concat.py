'''
Description:
Version:
Author: Leidi
Date: 2021-10-27 14:05:30
LastEditors: Leidi
LastEditTime: 2022-02-25 10:39:39
'''
import argparse
import os
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def main(input_path, output_path, concate_num):

    image_input_folder = Path(input_path)
    image_output_folder = Path(output_path)
    concate_list = []
    temp_four_image = []
    for n, image_path in enumerate(image_input_folder.iterdir()):
        temp_four_image.append(image_path)
        if n % 4 == 0:
            concate_list.append(temp_four_image)
            temp_four_image = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='image_concat.py')
    parser.add_argument('--config', '--c', dest='config', default=r'Clean_up/config/default.yaml',
                        type=str, help='dataset config file path')
    parser.add_argument('--workers', '--w', dest='workers', default=32,
                        type=int, help='maximum number of dataloader workers(multiprocessing.cpu_count())')

    opt = parser.parse_args()
    # load dataset config file
    config = yaml.load(
        open(opt.config, 'r', encoding="utf-8"), Loader=yaml.FullLoader)

    main(config['input_folder'],
         config['output_folder'],
         config['concate_num'])
