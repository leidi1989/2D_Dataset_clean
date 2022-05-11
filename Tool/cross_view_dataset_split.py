'''
Description: 
Version: 
Author: Leidi
Date: 2022-05-04 20:07:30
LastEditors: Leidi
LastEditTime: 2022-05-11 11:04:38
'''
import argparse
import os
import random

import tqdm


def main(opt):
    
    file_path = opt.data_path + os.sep + 'input/'
    file_list = os.listdir(file_path)
    train_file_path = os.path.join(opt.output_path, 'train_files.txt')
    val_file_path = os.path.join(opt.output_path, 'val_files.txt')
    fh_train = open(train_file_path, 'w')
    fh_val = open(val_file_path, 'w')
    num_train = len(file_list) * (1 - float(opt.val_ratio))
    
    for n, file_name in enumerate(file_list):
        if n < num_train:
            fh_train.write(''.join(file_path+file_name))
            if n != num_train - 1:
                fh_train.write('\n')
        else:
            fh_val.write(''.join(file_path+file_name)) 
            if n != len(file_list) - 1:
                fh_val.write('\n')
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='clean.py')
    parser.add_argument("--data_path", type=str, 
                        default='/mnt/data_2/Dataset/Autopilot_bev_dataset/hy_bev_multi_80_80_80_80_20220426_20220427_20220509_cross_view',
                        help="path to folder of raw images")
    parser.add_argument("--val_ratio", type=float, 
                        default=0.2,
                        help="ratio of validation and train data")
    parser.add_argument("--output_path", type=str, 
                        default='/mnt/data_2/Dataset/Autopilot_bev_dataset/hy_bev_multi_80_80_80_80_20220426_20220427_20220509_cross_view',
                        help="path to folder of raw images")

    opt = parser.parse_args()

    main(opt)
