'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:13:19
LastEditors: Leidi
LastEditTime: 2022-01-18 14:12:29
'''
import os
import cv2
import json
import numpy as np


class BOX:
    """真实框类"""

    def __init__(self,
                 box_clss: str,
                 box_xywh: list,
                 box_color: str = '',
                 box_tool: str = '',
                 box_difficult: int = 0,
                 box_distance: float = 0,
                 box_occlusion: float = 0
                 ) -> None:
        """[真实框类]

        Args:
            box_clss (str): [类别]
            xywh (list): [bbox左上点和宽高]
            color (str, optional): [真实框目标颜色]. Defaults to ''.
            tool (str, optional): [bbox工具]. Defaults to ''.
            difficult (int, optional): [困难样本]. Defaults to 0.
            distance (float, optional): [真实框中心点距离]. Defaults to 0.
            occlusion (float, optional): [真实框遮挡率]. Defaults to 0.
        """

        self.box_clss = box_clss
        self.box_xywh = box_xywh
        self.box_color = box_color
        self.box_tool = box_tool
        self.box_difficult = box_difficult
        self.box_distance = box_distance
        self.box_occlusion = box_occlusion

    def box_get_area(self) -> float:
        """[获取box面积]

        Returns:
            float: [box面积]
        """

        return self.box_xywh[2] * self.box_xywh[3]


class SEGMENTATION:
    """真分割类"""

    def __init__(self,
                 segmentation_clss: str,
                 segmentation: list,
                 segmentation_area: int = None,
                 segmentation_iscrowd: int = 0,
                 ) -> None:
        """[真分割]

        Args:
            segmentation_clss (str): [类别]
            segmentation (list): [分割区域列表]
            area (float, optional): [像素区域大小]. Defaults to 0.
            iscrowd (int, optional): [是否使用coco2017中的iscrowd格式]. Defaults to 0.
        """

        self.segmentation_clss = segmentation_clss
        self.segmentation = segmentation
        if segmentation_area == None:
            self.segmentation_area = int(
                cv2.contourArea(np.array(self.segmentation)))
        else:
            self.segmentation_area = segmentation_area
        self.segmentation_iscrowd = int(segmentation_iscrowd)

    def segmentation_get_bbox_area(self):

        segmentation = np.asarray(self.segmentation)
        min_x = int(np.min(segmentation[:, 0]))
        min_y = int(np.min(segmentation[:, 1]))
        max_x = int(np.max(segmentation[:, 0]))
        max_y = int(np.max(segmentation[:, 1]))
        box_xywh = [min_x, min_y, max_x-min_x, max_y-min_y]

        return box_xywh[2] * box_xywh[3]


class KEYPOINTS:
    """真实关键点类"""

    def __init__(self,
                 keypoints_clss: str,
                 keypoints_num: int,
                 keypoints: list
                 ) -> None:
        """[真实关键点类]

        Args:
            clss (str): [类别]
            num_keypoints (int): [关键点数量]
            keypoints (list): [关键点坐标列表]
        """

        self.keypoints_clss = keypoints_clss
        self.keypoints_num = keypoints_num
        self.keypoints = keypoints


class OBJECT(BOX, SEGMENTATION, KEYPOINTS):
    """标注物体类"""

    def __init__(self,
                 object_id: int,
                 object_clss: str,
                 box_clss: str,
                 segmentation_clss: str,
                 keypoints_clss: str,

                 box_xywh: list,
                 segmentation: list,

                 keypoints_num: int,
                 keypoints: list,

                 box_color: str = '',
                 box_tool: str = '',
                 box_difficult: int = 0,
                 box_distance: float = 0,
                 box_occlusion: float = 0,

                 segmentation_area: int = None,
                 segmentation_iscrowd: int = 0,
                 ) -> None:
        """[summary]

        Args:
            object_clss (str): [标注目标类别]
            box_clss (str): [真实框类别]
            segmentation_clss (str): [分割区域类别]
            keypoints_clss (str): [关键点类别]
            xywh (list): [真实框x，y，width，height列表]
            segmentation (list): [分割多边形点列表]
            num_keypoints (int): [关键点个数]
            keypoints (list): [关键点坐标]
            box_color (str, optional): [真实框颜色]. Defaults to ''.
            box_tool (str, optional): [真实框标注工具]. Defaults to ''.
            box_difficult (int, optional): [真实框困难程度]. Defaults to 0.
            box_distance (float, optional): [真实框距离]. Defaults to 0.
            box_occlusion (float, optional): [真实框遮挡比例]. Defaults to 0.
            segmentation_area (int, optional): [分割区域像素大小]. Defaults to None.
            segmentation_iscrowd (int, optional): [是否使用coco2017中的iscrowd格式]. Defaults to 0.
        """

        BOX.__init__(self, box_clss, box_xywh,
                     box_color=box_color, box_tool=box_tool, box_difficult=box_difficult,
                     box_distance=box_distance, box_occlusion=box_occlusion)
        SEGMENTATION.__init__(self, segmentation_clss, segmentation,
                              segmentation_area=segmentation_area, segmentation_iscrowd=segmentation_iscrowd)
        KEYPOINTS.__init__(self, keypoints_clss, keypoints_num, keypoints)
        self.object_id = object_id
        self.object_clss = object_clss


class IMAGE:
    """图片类"""

    def __init__(self,
                 image_name_in: str,
                 image_name_new_in: str,
                 image_path_in: str,
                 height_in: int,
                 width_in: int,
                 channels_in: int,
                 object_list_in: list,
                 ) -> None:
        """[图片类]

        Args:
            image_name_in (str): [图片名称]
            image_name_new_in (str): [图片新名称]
            image_path_in (str): [图片路径]
            height_in (int): [图片高]
            width_in (int): [图片宽]
            channels_in (int): [图片通道数]
            object_list_in (list): [标注目标列表]
        """

        self.image_name = image_name_in                     # 图片名称
        self.image_name_new = image_name_new_in             # 修改后图片名称
        self.file_name = os.path.splitext(self.image_name)[0]
        self.file_name_new = os.path.splitext(self.image_name_new)[0]
        self.image_path = image_path_in                     # 图片地址
        self.height = height_in                             # 图片高
        self.width = width_in                               # 图片宽
        self.channels = channels_in                         # 图片通道数
        self.object_list = object_list_in

    def modify_object_list(self, input_dataset: object) -> None:
        """[修改真实框类别]

        Args:
            input_dataset (object): [输入数据集实例]
        """
        for task, task_class_dict in input_dataset.task_dict.items():
            if task_class_dict['Modify_class_dict'] is not None:
                for one_object in self.object_list:
                    # 遍历融合类别文件字典，完成label中的类别修改，
                    # 若此bbox类别属于混合标签类别列表，则返回该标签在混合类别列表的索引值，修改类别。
                    if task == 'Detection':
                        for (key, value) in task_class_dict['Modify_class_dict'].items():
                            if one_object.box_clss in set(value):
                                one_object.box_clss = key
                    elif task == 'Semantic_segmentation':
                        for (key, value) in task_class_dict['Modify_class_dict'].items():
                            if one_object.segmentation_clss in set(value):
                                one_object.segmentation_clss = key
                    elif task == 'Instance_segmentation':
                        for (key, value) in task_class_dict['Modify_class_dict'].items():
                            if one_object.box_clss in set(value):
                                one_object.box_clss = key
                            if one_object.segmentation_clss in set(value):
                                one_object.segmentation_clss = key
                    elif task == 'Keypoint':
                        for (key, value) in task_class_dict['Modify_class_dict'].items():
                            if one_object.keypoints_class in set(value):
                                one_object.keypoints_class = key

        return

    def object_pixel_limit(self, input_dataset: object) -> None:
        """[对标注目标进行像素大小筛选]

        Args:
            input_dataset (object): [数据集实例]
        """

        for task, task_class_dict in input_dataset.task_dict.items():
            if task_class_dict['Target_object_pixel_limit_dict'] is not None:
                for n, object in enumerate(self.object_list):
                    if task == 'Detection' or task == 'Instance_segmentation' or task == 'Keypoint':
                        pixel = object.box_get_area()
                        if pixel < task_class_dict['Target_object_pixel_limit_dict'][object.box_clss][0] or \
                                pixel > task_class_dict['Target_object_pixel_limit_dict'][object.box_clss][1]:
                            self.object_list.pop(self.object_list.index(n))
                    elif task == 'Semantic_segmentation':
                        pixel = object.segmentation_get_bbox_area()
                        if pixel < task_class_dict['Target_object_pixel_limit_dict'][object.segmentation_clss][0] or \
                                pixel > task_class_dict['Target_object_pixel_limit_dict'][object.segmentation_clss][1]:
                            self.object_list.pop(self.object_list.index(n))

        return

    def output_temp_annotation(self, temp_annotation_output_path):
        """[输出temp dataset annotation]

        Args:
            annotation_output_path (str): [temp dataset annotation输出路径]

        Returns:
            bool: [输出是否成功]
        """

        if self == None:
            return False

        annotation = {'name': self.file_name_new,
                      'frames': [{'timestamp': 10000,
                                  'objects': []}],
                      'attributes': {'weather': 'undefined',
                                     'scene': 'city street',
                                     'timeofday': 'daytime'
                                     }
                      }
        for object in self.object_list:
            # 真实框
            object = {'id': object.object_id,
                      'object_clss': object.object_clss,
                      'box_clss': object.box_clss,
                      'box_color': object.box_color,
                      'box_difficult': object.box_difficult,
                      'box_distance': object.box_distance,
                      'box_occlusion': object.box_occlusion,
                      'box_tool': object.box_tool,
                      'box_xywh': object.box_xywh,
                      'keypoints_clss': object.keypoints_clss,
                      'keypoints_num': object.keypoints_num,
                      'keypoints': object.keypoints,
                      'segmentation_clss': object.segmentation_clss,
                      'segmentation': object.segmentation,
                      'segmentation_area': object.segmentation_area,
                      'segmentation_iscrowd': object.segmentation_iscrowd
                      }
            annotation['frames'][0]['objects'].append(object)
        # 输出json文件
        json.dump(annotation, open(temp_annotation_output_path, 'w'))

        return True
