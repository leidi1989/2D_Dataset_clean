'''
Description:
Version:
Author: Leidi
Date: 2021-08-09 16:05:57
LastEditors: Leidi
LastEditTime: 2021-12-31 17:04:31
'''
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from base.image_base import IMAGE
matplotlib.rc("font", family='AR PL UMing CN')
plt.switch_backend('agg')


def plot_sample_statistics(dataset) -> None:
    """[绘制样本统计图]

    Args:
        dataset ([数据集类]): [数据集类实例]
    """
    if 'unlabeled' in dataset['class_list_new']:
        x = np.arange(len(dataset['class_list_new']))  # x为类别数量
    else:
        x = np.arange(len(dataset['class_list_new']) + 1)  # x为类别数量
    fig = plt.figure(1, figsize=(
        len(dataset['class_list_new']), 9))   # 图片宽比例为类别个数

    # 绘图
    # 绘制真实框数量柱状图
    ax = fig.add_subplot(211)   # 单图显示类别计数柱状图
    ax.set_title('Dataset distribution',
                 bbox={'facecolor': '0.8', 'pad': 2})
    # width_list = [-0.45, -0.15, 0.15, 0.45]
    width_list = [0, 0, 0, 0, 0]
    colors = ['dodgerblue', 'aquamarine',
              'pink', 'mediumpurple', 'slategrey']

    print('Plot bar chart:')
    for one_set_label_path_list, set_size, clrs in tqdm(zip(dataset['temp_divide_count_dict_list'],
                                                            width_list, colors),
                                                        total=len(dataset['temp_divide_count_dict_list'])):
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
            for m, b in zip(x, values):     # 为柱状图添加标签
                plt.text(m + set_size, b, '%.0f' %
                         b, ha='center', va='bottom', fontsize=10)
        if colors.index(clrs) == 1:
            for m, b in zip(x, values):     # 为柱状图添加标签
                plt.text(m + set_size, b, '%.0f' %
                         b, ha='center', va='top', fontsize=10, color='r')
        plt.xticks(x, labels, rotation=45)      # 使x轴标签逆时针倾斜45度
        plt.tight_layout()
        plt.legend(['Total', 'Train', 'val', 'test', 'redund'], loc='best')

    # 绘制占比点线图
    at = fig.add_subplot(212)   # 单图显示类别占比线条图
    at.set_title('Dataset proportion',
                 bbox={'facecolor': '0.8', 'pad': 2})
    width_list = [0, 0, 0, 0, 0]
    thread_type_list = ['*', '*--', '.-.', '+-.', '-']

    print('Plot linear graph:')
    for one_set_label_path_list, set_size, clrs, thread_type in tqdm(zip(dataset['temp_divide_proportion_dict_list'],
                                                                         width_list, colors, thread_type_list),
                                                                     total=len(dataset['temp_divide_proportion_dict_list'])):
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
    plt.legend(['Total', 'Train', 'val', 'test', 'redund'], loc='best')
    plt.savefig(os.path.join(dataset['temp_informations_folder'],
                             'Dataset distribution.tif'), bbox_inches='tight')
    # plt.show()
    plt.close(fig)

    return


def plot_true_box(dataset) -> None:
    """[绘制每张图片的真实框检测图]

    Args:
        dataset ([Dataset]): [Dataset类实例]
        image (IMAGE): [IMAGE类实例]
    """

    # 类别色彩
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(dataset['class_list_new']))]
    # 统计各个类别的框数
    nums = [[] for _ in range(len(dataset['class_list_new']))]
    image_count = 0
    plot_true_box_success = 0
    plot_true_box_fail = 0
    total_box = 0
    print('Output check true box annotation images:')
    for image in tqdm(dataset['check_images_list']):
        image_path = os.path.join(
            dataset['temp_images_folder'], image.image_name)
        output_image = cv2.imread(image_path)  # 读取对应标签图片
        for box in image.true_box_list:  # 获取每张图片的bbox信息
            try:
                nums[dataset['class_list_new'].index(
                    box.clss)].append(box.clss)
                color = colors[dataset['class_list_new'].index(
                    box.clss)]
                # if dataset['target_annotation_check_mask'] == False:
                cv2.rectangle(output_image, (int(box.xmin), int(box.ymin)),
                              (int(box.xmax), int(box.ymax)), color, thickness=2)
                plot_true_box_success += 1
                # 绘制透明锚框
                # else:
                #     zeros1 = np.zeros((output_image.shape), dtype=np.uint8)
                #     zeros1_mask = cv2.rectangle(zeros1, (box.xmin, box.ymin),
                #                                 (box.xmax, box.ymax),
                #                                 color, thickness=-1)
                #     alpha = 1   # alpha 为第一张图片的透明度
                #     beta = 0.5  # beta 为第二张图片的透明度
                #     gamma = 0
                #     # cv2.addWeighted 将原始图片与 mask 融合
                #     mask_img = cv2.addWeighted(
                #         output_image, alpha, zeros1_mask, beta, gamma)
                #     output_image = mask_img
                #     plot_true_box_success += 1

                # cv2.putText(output_image, box.clss, (int(box.xmin), int(box.ymin)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
            except:
                print(image.image_name + str(box.clss) + "is not in list")
                plot_true_box_fail += 1
                continue
            total_box += 1
            # 输出图片
        path = os.path.join(
            dataset['check_annotation_output_folder'], image.image_name)
        cv2.imwrite(path, output_image)
        image_count += 1

    # 输出检查统计
    print("\nTotal check annotations count: \t%d" % image_count)
    print('Check annotation true box count:')
    print("Plot true box success image: \t%d" % plot_true_box_success)
    print("Plot true box fail image:    \t%d" % plot_true_box_fail)
    print('True box class count:')
    for i in nums:
        if len(i) != 0:
            print(i[0] + ':' + str(len(i)))

    with open(os.path.join(dataset['check_annotation_output_folder'], 'detect_class_count.txt'), 'w') as f:
        for i in nums:
            if len(i) != 0:
                temp = i[0] + ':' + str(len(i)) + '\n'
                f.write(temp)
        f.close()

    return


def plot_true_segmentation(dataset: dict) -> None:
    """[绘制每张图片的真实分割检测图]

    Args:
        dataset (dict): [Dataset类实例]
    """

    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(dataset['class_list_new']))]   # 类别色彩
    # 统计各个类别的框数
    nums = [[] for _ in range(len(dataset['class_list_new']))]
    image_count = 0
    plot_true_box_success = 0
    plot_true_box_fail = 0
    total_box = 0
    print('Output check images:')
    for image in tqdm(dataset['check_images_list']):
        image_path = os.path.join(
            dataset['check_annotation_output_folder'], image.image_name)
        output_image = cv2.imread(image_path)  # 读取对应标签图片
        for object in image.true_segmentation_list:  # 获取每张图片的bbox信息
            # try:
            nums[dataset['class_list_new'].index(
                object.clss)].append(object.clss)
            class_color = colors[dataset['class_list_new'].index(
                object.clss)]
            if dataset['target_annotation_check_mask'] == False:
                points = np.array(object.segmentation)
                cv2.polylines(
                    output_image, pts=[points], isClosed=True, color=class_color, thickness=2)
                plot_true_box_success += 1
            # 绘制透明真实框
            else:
                zeros1 = np.zeros((output_image.shape), dtype=np.uint8)
                points = np.array(object.segmentation)
                zeros1_mask = cv2.fillPoly(
                    zeros1, pts=[points], color=class_color)
                alpha = 1   # alpha 为第一张图片的透明度
                beta = 0.5  # beta 为第二张图片的透明度
                gamma = 0
                # cv2.addWeighted 将原始图片与 mask 融合
                mask_img = cv2.addWeighted(
                    output_image, alpha, zeros1_mask, beta, gamma)
                output_image = mask_img
                plot_true_box_success += 1

            cv2.putText(output_image, object.clss, (int(object.segmentation[0][0]), int(object.segmentation[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
            total_box += 1
            # 输出图片
        path = os.path.join(
            dataset['check_annotation_output_folder'], image.image_name)
        cv2.imwrite(path, output_image)
        image_count += 1

    # 输出检查统计
    print("\nTotal check annotations count: \t%d" % image_count)
    print('Check annotation true box count:')
    print("Plot true segment success image: \t%d" % plot_true_box_success)
    print("Plot true segment fail image:    \t%d" % plot_true_box_fail)
    for i in nums:
        if len(i) != 0:
            print(i[0] + ':' + str(len(i)))

    with open(os.path.join(dataset['check_annotation_output_folder'], 'class_count.txt'), 'w') as f:
        for i in nums:
            if len(i) != 0:
                temp = i[0] + ':' + str(len(i)) + '\n'
                f.write(temp)
        f.close()

    return


def plot_segment_annotation(dataset: dict, image: IMAGE, segment_annotation_output_path: str) -> None:
    """[绘制分割标签图]

    Args:
        dataset (dict): [数据集信息字典]
        image (IMAGE): [图片类实例]
        segment_annotation_output_path (str): [分割标签图输出路径]
    """
    zeros = np.zeros((image.height, image.width), dtype=np.uint8)
    if len(image.true_segmentation_list):
        for seg in image.true_segmentation_list:
            class_color = dataset['segment_class_list_new'].index(
                seg.clss)
            points = np.array(seg.segmentation)
            zeros_mask = cv2.fillPoly(
                zeros, pts=[points], color=class_color)
            cv2.imwrite(segment_annotation_output_path, zeros_mask)
    else:
        cv2.imwrite(segment_annotation_output_path, zeros)

    return
