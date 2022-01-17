'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:13:19
LastEditors: Leidi
LastEditTime: 2022-01-17 16:24:12
'''
import os
import cv2
import json
import numpy as np


class BOX:
    """真实框类"""

    def __init__(self,
                 box_clss: str,
                 xywh: list,
                 color: str = '',
                 tool: str = '',
                 difficult: int = 0,
                 distance: float = 0,
                 occlusion: float = 0
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
        self.box_xywh = xywh
        self.color = color
        self.tool = tool
        self.difficult = difficult
        self.distance = distance
        self.occlusion = occlusion

    def get_box_area(self) -> float:
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
                 area: int = None,
                 iscrowd: int = 0,
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
        if area == None:
            self.segmentation_area = int(
                cv2.contourArea(np.array(self.segmentation)))
        else:
            self.segmentation_area = area
        self.iscrowd = int(iscrowd)

    def get_segmentation_bbox_area(self):

        segmentation = np.asarray(self.segmentation)
        min_x = int(np.min(segmentation[:, 0]))
        min_y = int(np.min(segmentation[:, 1]))
        max_x = int(np.max(segmentation[:, 0]))
        max_y = int(np.max(segmentation[:, 1]))
        self.box_xywh = [min_x, min_y, max_x-min_x, max_y-min_y]

        return self.box_xywh[2] * self.box_xywh[3]


class KEYPOINTS:
    """真实关键点类"""

    def __init__(self,
                 keypoints_clss: str,
                 num_keypoints: int,
                 keypoints: list
                 ) -> None:
        """[真实关键点类]

        Args:
            clss (str): [类别]
            num_keypoints (int): [关键点数量]
            keypoints (list): [关键点坐标列表]
        """

        self.keypoints_clss = keypoints_clss
        self.num_keypoints = num_keypoints
        self.keypoints = keypoints


class OBJECT(BOX, SEGMENTATION, KEYPOINTS):
    """标注物体类"""

    def __init__(self,
                 object_clss: str,
                 box_clss: str,
                 segmentation_clss: str,
                 keypoints_clss: str,

                 xywh: list,
                 segmentation: list,

                 num_keypoints: int,
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

        BOX.__init__(self, box_clss, xywh,
                     color=box_color, tool=box_tool, difficult=box_difficult,
                     distance=box_distance, occlusion=box_occlusion)
        SEGMENTATION.__init__(self, segmentation_clss, segmentation,
                              area=segmentation_area, iscrowd=segmentation_iscrowd)
        KEYPOINTS.__init__(self, keypoints_clss, num_keypoints, keypoints)
        self.object_clss = object_clss

    def get_outer_bbox(self):
        """[将分割按最外围矩形框转换为bbox]
        """

        segmentation = np.asarray(self.segmentation)
        min_x = int(np.min(segmentation[:, 0]))
        min_y = int(np.min(segmentation[:, 1]))
        max_x = int(np.max(segmentation[:, 0]))
        max_y = int(np.max(segmentation[:, 1]))
        self.box_xywh = [min_x, min_y, max_x-min_x, max_y-min_y]


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
                        pixel = object.get_box_area()
                        if pixel < task_class_dict['Target_object_pixel_limit_dict'][object.box_clss][0] or \
                                pixel > task_class_dict['Target_object_pixel_limit_dict'][object.box_clss][1]:
                            self.object_list.pop(self.object_list.index(n))
                    elif task == 'Semantic_segmentation':
                        pixel = object.get_segmentation_bbox_area()
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

        # 真实框
        for i, n in enumerate(self.true_box_list):
            box = {'category': n.clss,
                   'id': i,
                   'attributes': {'occluded': False if 0 == n.difficult else str(n.difficult),
                                  'truncated': False if 0 == n.occlusion else str(n.occlusion),
                                  'trafficLightColor': "none"
                                  },
                   'box2d': {'x1': int(n.xmin),
                             'y1': int(n.ymin),
                             'x2': int(n.xmax),
                             'y2': int(n.ymax),
                             }
                   }
            annotation['frames'][0]['objects'].append(box)

        # 语义分割
        m = len(self.true_box_list)
        for i, n in enumerate(self.true_segmentation_list):
            segmentation = {'category': n.clss,
                            'id': i + m,
                            'attributes': {},
                            'poly2d': n.segmentation
                            }
            annotation['frames'][0]['objects'].append(segmentation)

        # 关键点
        l = len(self.true_segmentation_list)
        for i, n in enumerate(self.true_keypoint_list):
            keypoint = {'category': n.clss,
                        'id': i + m + l,
                        'attributes': {},
                        'num_keypoints': n.num_keypoints,
                        'keypoint': n.keypoints
                        }
            annotation['frames'][0]['objects'].append(keypoint)

        # 输出json文件
        json.dump(annotation, open(temp_annotation_output_path, 'w'))

        return True
