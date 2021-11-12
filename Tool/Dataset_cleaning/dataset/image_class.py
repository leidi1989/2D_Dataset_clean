'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-28 15:44:07
LastEditors: Leidi
LastEditTime: 2021-04-28 15:44:44
'''
import numpy as np


class true_box:
    """真实框类"""

    def __init__(self, clss, xmin, ymin, xmax, ymax, color='', tool='', difficult=0, distance=0, occlusion=0):
        self.clss = clss
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.color = color  # 真实框目标颜色
        self.tool = tool    # bbox工具
        self.difficult = difficult
        self.distance = distance    # 真实框中心点距离
        self.occlusion = occlusion    # 真实框遮挡率


class per_image:
    """图片类"""

    def __init__(self, image_name_in, image_path_in, height_in, width_in, channels_in, true_box_list_in):
        self.image_name = image_name_in    # 图片名称
        self.image_path = image_path_in    # 图片地址
        self.height = height_in    # 图片高
        self.width = width_in  # 图片宽
        self.channels = channels_in    # 图片通道数
        self.true_box_list = true_box_list_in  # 图片真实框列表
        self.free_area = len(self.free_space_area())    # 获取图片掩码图的非真实框面积

    def true_box_list_updata(self, one_bbox_data):
        """[为per_image对象true_box_list成员添加元素]

        Parameters
        ----------
        one_bbox_data : [class true_box]
            [真实框类]
        """

        self.true_box_list.append(one_bbox_data)

    def get_true_box_mask(self):
        """[获取图片真实框掩码图，前景置1，背景置0]

        Parameters
        ----------
        true_box_list : [list]
            [真实框列表]
        """

        image_mask = np.zeros([self.height, self.width])
        for one_ture_box in self.true_box_list:     # 读取true_box并对前景在mask上置1
            image_mask[int(one_ture_box.ymin):int(one_ture_box.ymax),
                       int(one_ture_box.xmin):int(one_ture_box.xmax)] = 1     # 将真实框范围内置1

        return image_mask

    # TODO
    def free_space_area(self):
        """[获取图片非真实框像素列表]

        Returns
        -------
        free_space_area_list : [list]
            [图片非真实框像素列表]
        """

        mask_true_box = self.get_true_box_mask()

        return np.argwhere(mask_true_box == 0)