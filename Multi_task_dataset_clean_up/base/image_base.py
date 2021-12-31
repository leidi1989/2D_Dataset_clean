'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:13:19
LastEditors: Leidi
LastEditTime: 2021-12-31 16:30:24
'''
import os
import cv2
import numpy as np


class TRUE_BOX:
    """真实框类"""

    def __init__(self,
                 clss: str,
                 xmin: float,
                 ymin: float,
                 xmax: float,
                 ymax: float,
                 color: str = '',
                 tool: str = '',
                 difficult: int = 0,
                 distance: float = 0,
                 occlusion: float = 0
                 ) -> None:
        """[真实框类]

        Args:
            clss (str): [description]
            xmin (float): [description]
            ymin (float): [description]
            xmax (float): [description]
            ymax (float): [description]
            color (str, optional): [description]. Defaults to ''.
            tool (str, optional): [description]. Defaults to ''.
            difficult (int, optional): [description]. Defaults to 0.
            distance (float, optional): [description]. Defaults to 0.
            occlusion (float, optional): [description]. Defaults to 0.
        """

        self.clss = clss
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.color = color              # 真实框目标颜色
        self.tool = tool                # bbox工具
        self.difficult = difficult      # 困难样本
        self.distance = distance        # 真实框中心点距离
        self.occlusion = occlusion      # 真实框遮挡率


class TRUE_SEGMENTATION:
    """真分割类"""

    def __init__(self,
                 clss: str,
                 segmentation: list,
                 segmentation_bounding_box: list = None,
                 area: int = None,
                 iscrowd: int = 0,
                 ) -> None:
        """[真分割]

        Args:
            clss (str): [类别]
            segmentation (list): [分割区域列表]
            area (float, optional): [像素区域大小]. Defaults to 0.
            iscrowd (int, optional): [是否拥挤]. Defaults to 0.
        """

        self.clss = clss
        self.segmentation = segmentation
        if segmentation_bounding_box == None:
            self.segmentation_bounding_box = self.get_outer_bounding_box()
        else:
            self.segmentation_bounding_box = segmentation_bounding_box
        if area == None:
            self.area = int(cv2.contourArea(np.array(self.segmentation)))
        else:
            self.area = area
        self.iscrowd = int(iscrowd)

    def get_outer_bounding_box(self):
        """[将分割按最外围矩形框转换为bbox]

        Args:
            segmentation (list): [真实分割]

        Returns:
            list: [转换后真实框左上点右下点坐标]
        """

        segmentation = np.asarray(self.segmentation)
        min_x = np.min(segmentation[:, 0])
        min_y = np.min(segmentation[:, 1])
        max_x = np.max(segmentation[:, 0])
        max_y = np.max(segmentation[:, 1])
        bbox = [int(min_x), int(min_y), int(max_x), int(max_y)]

        return bbox


class IMAGE:
    """图片类"""

    def __init__(self,
                 image_name_in: str,
                 image_name_new_in: str,
                 image_path_in: str,
                 height_in: int,
                 width_in: int,
                 channels_in: int,
                 true_box_list_in: list,
                 true_segmentation_list_in: list,
                 ) -> None:
        """[图片类]

        Args:
            image_name_in (str): [图片名称]
            image_name_new_in (str): [图片新名称]
            image_path_in (str): [图片路径]
            height_in (int): [图片高]
            width_in (int): [图片宽]
            channels_in (int): [图片通道数]
            true_box_list_in (list): [真实框列表]
            true_segmentation_list_in (list): [真实分割列表]
        """

        self.image_name = image_name_in                     # 图片名称
        self.image_name_new = image_name_new_in             # 修改后图片名称
        self.file_name = os.path.splitext(self.image_name)[0]
        self.file_name_new = os.path.splitext(self.image_name_new)[0]
        self.image_path = image_path_in                     # 图片地址
        self.height = height_in                             # 图片高
        self.width = width_in                               # 图片宽
        self.channels = channels_in                         # 图片通道数
        self.true_box_list = true_box_list_in               # 图片真实框列表
        self.true_segmentation_list = true_segmentation_list_in    # 图片真实分割列表

    def true_box_list_updata(self, one_bbox_data: TRUE_BOX) -> None:
        """[为per_image对象true_box_list成员添加元素]

        Args:
            one_bbox_data (true_box): [TRUE_BOX类真实框变量]
        """

        self.true_box_list.append(one_bbox_data)

    def true_segmentation_list_updata(self, one_segmentation_data: TRUE_SEGMENTATION) -> None:
        """[为per_image对象true_segementation_list成员添加元素]

        Args:
            one_segmentation_data (true_segmentation): [TRUE_SEGMENTATION类真实框变量]
        """

        self.true_segmentation_list.append(one_segmentation_data)

    def segmentation_create_box(self) -> None:
        """[使用分割信息创建真实框信息]
        """

        for n in self.true_segmentation_list:
            x_y = np.array(n.segmentation)
            x_min = np.min(x_y, axis=0)[0]
            x_max = np.max(x_y, axis=0)[0]
            y_min = np.min(x_y, axis=0)[1]
            y_max = np.max(x_y, axis=0)[1]
            self.true_box_list_updata(
                TRUE_BOX(n.clss, x_min, y_min, x_max, y_max))

        return
