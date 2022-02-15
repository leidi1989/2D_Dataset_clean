'''
Description: 
Version: 
Author: Leidi
Date: 2022-02-15 13:51:06
LastEditors: Leidi
LastEditTime: 2022-02-15 13:52:10
'''
from utils.utils import *

import os
import math
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='AR PL UMing CN')
plt.switch_backend('agg')


def plot_detection_sample_statistics(self, task: str, task_class_dict: dict) -> None:
    """[绘制detection样本统计图]

    Args:
        task (str): [任务类型]
        task_class_dict (dict): [任务类别字典]
    """

    x = np.arange(len(task_class_dict['Target_dataset_class']))  # x为类别数量
    fig = plt.figure(1, figsize=(
        len(task_class_dict['Target_dataset_class']), 9))   # 图片宽比例为类别个数

    # 绘图
    # 绘制真实框数量柱状图
    ax = fig.add_subplot(211)   # 单图显示类别计数柱状图
    ax.set_title('Dataset distribution',
                 bbox={'facecolor': '0.8', 'pad': 2})
    # width_list = [-0.45, -0.15, 0.15, 0.45]
    width_list = [0, 0, 0, 0, 0]
    colors = ['dodgerblue', 'aquamarine',
              'pink', 'mediumpurple', 'slategrey']
    bar_width = 0
    print('Plot bar chart:')
    for one_set_label_path_list, set_size, clrs in tqdm(zip(self.temp_divide_count_dict[task].values(),
                                                            width_list, colors),
                                                        total=len(self.temp_divide_count_dict[task].values())):
        labels = []     # class
        values = []     # class count
        # 遍历字典分别将键名和对应的键值存入绘图标签列表、绘图y轴列表中
        # for key, value in sorted(one_set_label_path_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
        for key, value in one_set_label_path_list.items():
            labels.append(str(key))
            values.append(int(value))
            bar_width = max(bar_width, int(
                math.log10(value) if 0 != value else 1))
        # 绘制数据集类别数量统计柱状图
        ax.bar(x + set_size, values,
               width=1, color=clrs)
        if colors.index(clrs) == 0:
            for m, b in zip(x, values):     # 为柱状图添加标签
                plt.text(m + set_size, b, '%.0f' %
                         b, ha='center', va='bottom', fontsize=10)
        if colors.index(clrs) == 1:
            for m, b in zip(x, values):     # 为柱状图添加标签
                plt.text(m + set_size, b, '%.0f' %
                         b, ha='center', va='top', fontsize=10, color='r')
        plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                            top=0.8, wspace=0.3, hspace=0.2)
        plt.tight_layout()
    plt.legend(['Total', 'Train', 'val', 'test', 'redund'],
               loc='best', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # 绘制占比点线图
    at = fig.add_subplot(212)   # 单图显示类别占比线条图
    at.set_title('Dataset proportion',
                 bbox={'facecolor': '0.8', 'pad': 2})
    width_list = [0, 0, 0, 0, 0]
    thread_type_list = ['*', '*--', '.-.', '+-.', '-']

    print('Plot linear graph:')
    for one_set_label_path_list, set_size, clrs, thread_type in tqdm(zip(self.temp_divide_proportion_dict[task].values(),
                                                                         width_list, colors, thread_type_list),
                                                                     total=len(self.temp_divide_proportion_dict[task].values())):
        labels = []     # class
        values = []     # class count
        # 遍历字典分别将键名和对应的键值存入绘图标签列表、绘图y轴列表中
        # for key, value in sorted(one_set_label_path_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
        for key, value in one_set_label_path_list.items():
            labels.append(str(key))
            values.append(float(value))
        # 绘制数据集类别占比点线图状图
        at.plot(x, values, thread_type, linewidth=2, color=clrs)
        if colors.index(clrs) == 0:
            for m, b in zip(x, values):     # 为图添加标签
                plt.text(m + set_size, b, '%.2f%%' %
                         b, ha='center', va='bottom', fontsize=10)
        if colors.index(clrs) == 1:
            for m, b in zip(x, values):     # 为图添加标签
                plt.text(m + set_size, b, '%.2f%%' %
                         b, ha='center', va='top', fontsize=10, color='r')
        plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                            top=0.8, wspace=0.3, hspace=0.2)
        plt.tight_layout()
    plt.legend(['Total', 'Train', 'val', 'test', 'redund'],
               loc='best', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    plt.savefig(os.path.join(self.temp_sample_statistics_folder,
                             'Detection dataset distribution.tif'), bbox_inches='tight')
    # plt.show()
    plt.close(fig)

    return


def plot_segmentation_sample_statistics(self, task, task_class_dict) -> None:
    """[绘制segmentation样本统计图]

    Args:
        task (str): [任务类型]
        task_class_dict (dict): [任务类别字典]
    """

    if 'unlabeled' in task_class_dict['Target_dataset_class']:
        x = np.arange(
            len(task_class_dict['Target_dataset_class']))  # x为类别数量
    else:
        x = np.arange(
            len(task_class_dict['Target_dataset_class']) + 1)  # x为类别数量
    fig = plt.figure(1, figsize=(
        len(task_class_dict['Target_dataset_class']), 9))   # 图片宽比例为类别个数

    # 绘图
    # 绘制真实框数量柱状图
    ax = fig.add_subplot(211)   # 单图显示类别计数柱状图
    ax.set_title('Dataset distribution',
                 bbox={'facecolor': '0.8', 'pad': 2})
    # width_list = [-0.45, -0.15, 0.15, 0.45]
    width_list = [0, 0, 0, 0, 0]
    colors = ['dodgerblue', 'aquamarine',
              'pink', 'mediumpurple', 'slategrey']
    bar_width = 0

    print('Plot bar chart.')
    for one_set_label_path_list, set_size, clrs in \
        zip(self.temp_divide_count_dict[task].values(),
            width_list, colors):
        labels = []     # class
        values = []     # class count
        # 遍历字典分别将键名和对应的键值存入绘图标签列表、绘图y轴列表中
        # for key, value in sorted(one_set_label_path_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
        for key, value in one_set_label_path_list.items():
            labels.append(str(key))
            values.append(int(value))
            bar_width = max(bar_width, int(
                math.log10(value) if 0 != value else 1))
        # 绘制数据集类别数量统计柱状图
        ax.bar(x + set_size, values,
               width=1, color=clrs)
        if colors.index(clrs) == 0:
            for m, b in zip(x, values):     # 为柱状图添加标签
                plt.text(m + set_size, b, '%.0f' %
                         b, ha='center', va='bottom', fontsize=10)
        if colors.index(clrs) == 1:
            for m, b in zip(x, values):     # 为柱状图添加标签
                plt.text(m + set_size, b, '%.0f' %
                         b, ha='center', va='top', fontsize=10, color='r')
        plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                            top=0.8, wspace=0.3, hspace=0.2)
        plt.tight_layout()
    plt.legend(['Total', 'Train', 'val', 'test', 'redund'],
               loc='best', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # 绘制占比点线图
    at = fig.add_subplot(212)   # 单图显示类别占比线条图
    at.set_title('Dataset proportion',
                 bbox={'facecolor': '0.8', 'pad': 2})
    width_list = [0, 0, 0, 0, 0]
    thread_type_list = ['*', '*--', '.-.', '+-.', '-']

    print('Plot linear graph.')
    for one_set_label_path_list, set_size, clrs, thread_type \
        in zip(self.temp_divide_proportion_dict[task].values(),
               width_list, colors, thread_type_list):
        labels = []     # class
        values = []     # class count
        # 遍历字典分别将键名和对应的键值存入绘图标签列表、绘图y轴列表中
        # for key, value in sorted(one_set_label_path_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
        for key, value in one_set_label_path_list.items():
            labels.append(str(key))
            values.append(float(value))
        # 绘制数据集类别占比点线图状图
        at.plot(x, values, thread_type, linewidth=2, color=clrs)
        if colors.index(clrs) == 0:
            for m, b in zip(x, values):     # 为图添加标签
                plt.text(m + set_size, b, '%.2f%%' %
                         b, ha='center', va='bottom', fontsize=10)
        if colors.index(clrs) == 1:
            for m, b in zip(x, values):     # 为图添加标签
                plt.text(m + set_size, b, '%.2f%%' %
                         b, ha='center', va='top', fontsize=10, color='r')
        plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8,
                            top=0.8, wspace=0.3, hspace=0.2)
        plt.tight_layout()
    plt.legend(['Total', 'Train', 'val', 'test', 'redund'],
               loc='best', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    plt.savefig(os.path.join(self.temp_sample_statistics_folder,
                             'Semantic segmentation dataset distribution.tif'), bbox_inches='tight')
    # plt.show()
    plt.close(fig)

    return
