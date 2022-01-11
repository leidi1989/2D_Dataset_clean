'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:13:19
LastEditors: Leidi
LastEditTime: 2022-01-11 16:49:01
'''
import os
import cv2
import json
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


class TRUE_KEYPOINTS():
    """真实关键点类"""

    def __init__(self, clss: str, num_keypoints: int, keypoints: list) -> None:
        """[真实关键点类]

        Args:
            clss (str): [类别]
            num_keypoints (int): [关键点数量]
            keypoints (list): [关键点坐标列表]
        """

        self.clss = clss
        self.num_keypoints = num_keypoints
        self.keypoints = keypoints


class TRUE_OBJECT():
    """标注物体类"""

    def __init__(self,
                 true_box_in: TRUE_BOX = None,
                 true_segmentation_in: TRUE_SEGMENTATION = None,
                 true_keypoints_in: TRUE_KEYPOINTS = None) -> None:
        """[标注物体类]

        Args:
            true_box_in (TRUE_BOX): [真实框]
            true_segmentation_in (TRUE_SEGMENTATION): [真分割]
            true_keypoints_in (TRUE_KEYPOINTS): [真实关键点]
        """

        self.true_box = true_box_in
        self.true_segmentation = true_segmentation_in
        self.true_keypoints = true_keypoints_in


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
                 true_box_list_in: list,
                 true_segmentation_list_in: list,
                 true_keypoint_list_in: list
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
        self.object_list = object_list_in
        self.true_box_list = true_box_list_in               # 图片真实框列表
        self.true_segmentation_list = true_segmentation_list_in    # 图片真实分割列表
        self.true_keypoint_list = true_keypoint_list_in          # 图片真实关键点列表

    def true_object_list_updata(self, true_object_data: TRUE_OBJECT) -> None:
        """[为per_image对象true_box_list成员添加元素]

        Args:
            true_object_data (TRUE_OBJECT): [TRUE_OBJECT类变量]
        """

        self.object_list.append(true_object_data)

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

    def modify_true_box_list(self, class_modify_dict: dict) -> None:
        """[修改真实框类别]

        Args:
            class_modify_dict (dict): [类别修改字典]
        """

        if class_modify_dict is not None:
            for one_true_box in self.true_box_list:
                # 遍历融合类别文件字典，完成label中的类别修改，
                # 若此bbox类别属于混合标签类别列表，
                # 则返回该标签在混合类别列表的索引值
                for (key, value) in class_modify_dict.items():
                    if one_true_box.clss in set(value):
                        one_true_box.clss = key                     # 修改true_box类别

        return

    def modify_true_segmentation_list(self, class_modify_dict: dict) -> None:
        """[修改真实框类别]

        Args:
            class_modify_dict (dict): [类别修改字典]
        """

        if class_modify_dict is not None:
            for one_true_segmentation in self.true_segmentation_list:
                # 遍历融合类别文件字典，完成label中的类别修改，
                # 若此bbox类别属于混合标签类别列表，
                # 则返回该标签在混合类别列表的索引值
                for (key, value) in class_modify_dict.items():
                    if one_true_segmentation.clss in set(value):
                        one_true_segmentation.clss = key                     # 修改true_box类别

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
