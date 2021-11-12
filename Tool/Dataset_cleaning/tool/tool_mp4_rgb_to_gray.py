'''
Description: 
Version: 
Author: Leidi
Date: 2021-05-16 02:39:30
LastEditors: Leidi
LastEditTime: 2021-06-17 17:35:21
'''
from moviepy.editor import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cleaning_5_labels.py')
    parser.add_argument('--set', default=r'/mnt/disk1/detect_sample/telescope/20210616/img_20210616_2',
                        type=str, help='dataset path')
    parser.add_argument('--out', default=r'/mnt/disk1/detect_sample/telescope/20210616/img_20210616_3',
                        type=str, help='output path')
    opt = parser.parse_args()

    clip = VideoFileClip(opt.set)
    clipblackwhite = clip.fx(vfx.blackwhite)
    clipblackwhite.write_videofile(opt.out)
