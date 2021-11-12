'''
Description: 
Version: 
Author: Leidi
Date: 2021-02-22 03:58:53
LastEditors: Leidi
LastEditTime: 2021-05-15 17:49:59
'''
import os
from tqdm import tqdm

import sys
sys.path.append('/home/leidi/Workspace/2D_Dataset_clean/Dataset_cleaning')
from utils.utils import *


source_annotation_path = r'/media/leidi/My Passport/data/hy_dataset/CCTSDB/GroundTruth'


# total_classes = []
# for one_annotation in tqdm(os.listdir(source_annotation_path)):
#     with open(os.path.join(source_annotation_path, one_annotation), 'r') as f:
#         for one_line in f.readlines():
#             total_classes.append(one_line.split(' ')[0])

# classes_names = set(total_classes)
# for n in classes_names:
#     print(n)

# cctsdb
classes_output = check_out_file_exists(os.path.join(source_annotation_path, 'classes.names'))
total_classes = []
for one_annotation in tqdm(os.listdir(source_annotation_path)):
    with open(os.path.join(source_annotation_path, one_annotation), 'r') as f:
        for one_line in f.readlines():
            one = one_line.split(';')[-1].replace('\n', '').strip()
            total_classes.append(one)

classes_names = set(total_classes)
with open(classes_output, 'w') as f_out:
    for n in classes_names:
        if n != '':
            f_out.writelines(n + '\n')