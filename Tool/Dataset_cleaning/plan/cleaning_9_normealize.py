'''
Description: 
Version: 
Author: Leidi
Date: 2021-05-19 09:20:40
LastEditors: Leidi
LastEditTime: 2021-07-27 11:18:57
'''
'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-21 10:02:04
LastEditors: Leidi
LastEditTime: 2021-04-21 10:16:46
'''
# -*- coding:utf-8 -*-
import argparse
import os

from utils.utils import *


def normealize(output_path):
    """[summary]

    Args:
        output_path ([type]): [description]
    """    

    ImageSets_input_path = check_input_path(os.path.join(
        output_path, 'JPEGImages'))     # 获取数据集images路径
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cleaning_7_distribution.py')
    parser.add_argument('--out', default=r'',
                        type=str, help='output path')
    opt = parser.parse_args()

    output_path = check_output_path(opt.out)

    print('\nStart to normealize images：')
    normealize(output_path)
    print('Normealize images done!')
