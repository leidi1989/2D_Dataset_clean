# -*- coding:utf-8 -*-
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.utils import *

matplotlib.rc("font", family='AR PL UMing CN')

def get_path(output_path):
    """获取total.txt、train.txt、val.txt、test.txt绝对路径字典"""

    ttvt_path_list = []
    label_input_path = check_input_path(os.path.join(
        output_path, 'labels'))      # 获取数据集label路径
    ImageSets_input_path = check_input_path(os.path.join(
        output_path, 'ImageSets'))     # 获取数据集ImageSets路径
    total_input_path = check_input_path(os.path.join(
        ImageSets_input_path, 'total.txt'))      # 获取数据集total路径
    train_input_path = check_input_path(os.path.join(
        ImageSets_input_path, 'train.txt'))      # 获取数据集train路径
    val_input_path = check_input_path(os.path.join(
        ImageSets_input_path, 'val.txt'))      # 获取数据集val路径
    test_input_path = check_input_path(
        os.path.join(ImageSets_input_path, 'test.txt'))    # 获取数据集test路径

    ttvt_path_list = [total_input_path,
                      train_input_path, val_input_path, test_input_path]

    return ttvt_path_list, label_input_path


def get_one_set_label_path_list(ttvt_path_list):
    """获取每个set.txt文件下图片的标签地址列表"""
    every_set_label_list = []
    for one_txt in ttvt_path_list:
        # for one_txt_input_path in txt_input_path:
        with open(one_txt, 'r') as xx:
            one_set_labels_path = []
            for a in xx.readlines():
                ra = a.replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png', '.txt').strip('\n')
                one_set_labels_path.append(ra)
        every_set_label_list.append(one_set_labels_path)

    return every_set_label_list


def make_each_class_count_dict(label_input_path, every_set_label_list, class_list, ImageSets_input_path):
    """生成不同set的计数字典"""

    set_count_dict_list = []      # 声明set类别计数字典列表顺序为ttvt
    set_prop_dict_list = []      # 声明set类别计数字典列表顺序为ttvt
    set_name_list = ['total_distibution.txt', 'train_distibution.txt',
                     'val_distibution.txt', 'test_distibution.txt']

    for each_set_labels, set_name in zip(every_set_label_list, set_name_list):
        one_set_class_count_dict = {}   # 声明不同集的类别计数字典
        one_set_class_prop_dict = {}    # 声明不同集的类别占比字典
        for one_class in tqdm(class_list):
            one_set_class_count_dict[one_class] = 0     # 读取不同类别进计数字典作为键
            one_set_class_prop_dict[one_class] = float(0)   # 读取不同类别进占比字典作为键
        for one_label in each_set_labels:      # 统计全部labels各类别数量
            with open(os.path.join(label_input_path, one_label), 'r') as a_label:
                for b in a_label.readlines():
                    b = b.replace('\n', '')
                    # 获取对应class文件内bbox的类别
                    one_set_class_count_dict[str(
                        class_list[int(b.split(' ')[0])])] += 1
        set_count_dict_list.append(one_set_class_count_dict)

        one_set_total_count = 0     # 声明单数据集计数总数
        for _, value in one_set_class_count_dict.items():   # 计算数据集计数总数
            one_set_total_count += value
        for key, value in one_set_class_count_dict.items():
            one_set_class_prop_dict[key] = (float(value) / float(one_set_total_count)) * 100  # 计算个类别在此数据集占比
        set_prop_dict_list.append(one_set_class_prop_dict)

        with open(os.path.join(ImageSets_input_path, set_name), 'w') as dist_txt:    # 记录每个集的类别分布
            print('\n%s set class count:' % set_name.split('_')[0])
            for key, value in one_set_class_count_dict.items():
                dist_txt.write(str(key) + ':' + str(value) + '\n')
                print (str(key) + ':' + str(value))
            print('\n%s set porportion:' % set_name.split('_')[0])
            dist_txt.write('\n')
            for key, value in one_set_class_prop_dict.items():
                dist_txt.write(str(key) + ':' + str('%0.2f%%' % value) + '\n')
                print (str(key) + ':' + str('%0.2f%%' % value))

    return set_count_dict_list, set_prop_dict_list


def drow(set_count_dict_list, set_prop_dict_list, class_list, ImageSets_input_path):
    """在同图片中绘制不同set类别分布柱状图"""

    # 创建绘图底板
    x = np.arange(len(class_list))  # x为类别数量
    fig = plt.figure(1, figsize=(len(class_list), 9))   # 图片宽比例为类别个数

    # 绘制真实框数量柱状图
    ax = fig.add_subplot(211)   # 单图显示类别计数柱状图
    ax.set_title('Dataset distribution',
                 bbox={'facecolor': '0.8', 'pad': 2})
    # width_list = [-0.45, -0.15, 0.15, 0.45]
    width_list = [0, 0, 0, 0]
    colors = ['dodgerblue', 'aquamarine', 'pink', 'mediumpurple']
    # 遍历全部set的地址标签列表
    for one_set_label_path_list, set_size, clrs in zip(set_count_dict_list, width_list, colors):
        labels = []     # class
        values = []     # class count
        # 遍历字典分别将键名和对应的键值存入绘图标签列表、绘图y轴列表中
        # for key, value in sorted(one_set_label_path_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
        for key, value in one_set_label_path_list.items():
            labels.append(str(key))
            values.append(int(value))
        # 绘制数据集类别数量统计柱状图
        ax.bar(x + set_size, values, width=0.6, color=clrs)
        if colors.index(clrs) == 0:
            for a, b in zip(x, values):     # 为柱状图添加标签
                plt.text(a + set_size, b, '%.0f' %
                         b, ha='center', va='bottom', fontsize=10)
        if colors.index(clrs) == 1:
            for a, b in zip(x, values):     # 为柱状图添加标签
                plt.text(a + set_size, b, '%.0f' %
                         b, ha='center', va='top', fontsize=10, color='r')
        plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
        plt.tight_layout()
        plt.legend(['Total', 'Train', 'val', 'test'], loc='best')

    # 绘制占比点线图
    at = fig.add_subplot(212)   # 单图显示类别占比线条图
    at.set_title('Dataset proportion',
                 bbox={'facecolor': '0.8', 'pad': 2})
    width_list = [0, 0, 0, 0]
    colors = ['dodgerblue', 'aquamarine', 'pink', 'mediumpurple']
    thread_type_list = ['-','*--','.-.','+-.']
    # 遍历全部set的地址标签列表
    for one_set_label_path_list, set_size, clrs, thread_type in zip(set_prop_dict_list, width_list, colors, thread_type_list):
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
            for a, b in zip(x, values):     # 为图添加标签
                plt.text(a + set_size, b, '%.2f%%' %
                         b, ha='center', va='bottom', fontsize=10)
        if colors.index(clrs) == 1:
            for a, b in zip(x, values):     # 为图添加标签
                plt.text(a + set_size, b, '%.2f%%' %
                         b, ha='center', va='top', fontsize=10, color='r')
        plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
        plt.tight_layout()
    plt.legend(['Total', 'Train', 'val', 'test'], loc='best')

    plt.savefig(os.path.join(ImageSets_input_path,
                             'Dataset distribution.tif'), bbox_inches='tight')
    # plt.show()
