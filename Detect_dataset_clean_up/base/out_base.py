'''
Description:
Version:
Author: Leidi
Date: 2021-08-03 22:18:39
LastEditors: Leidi
LastEditTime: 2021-08-19 16:11:56
'''
from utils.utils import *
from base.image_base import IMAGE
from annotation.annotation_output import *
from utils.plot import plot_sample_statistics
from utils.convertion_function import revers_yolo
from base.dataset_characteristic import dataset_file_form
from annotation.annotation_output import annotation_output_funciton
from annotation.annotation_temp import TEMP_LOAD

import os
import cv2
import math
import shutil
import random
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='AR PL UMing CN')
plt.switch_backend('agg')


def out(dataset) -> None:
    """[输出annotation]
    """
    dataset['temp_annotation_path_list']= temp_annotation_path_list(dataset['temp_annotations_folder'])
    annotation_output_funciton(dataset['target_dataset_style'], dataset)
