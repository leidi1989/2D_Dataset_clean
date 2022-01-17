'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:13:19
LastEditors: Leidi
LastEditTime: 2022-01-17 11:30:19
'''
import os
import cv2
import json
import numpy as np


class BBOX:
    """真实框类"""

    def __init__(self,
                 bbox_clss: str,
                 xywh: list,
                 color: str = '',
                 tool: str = '',
                 difficult: int = 0,
                 distance: float = 0,
                 occlusion: float = 0
                 ) -> None:
        """[真实框类]

        Args:
            bbox_clss (str): [类别]
            xywh (list): [bbox左上点和宽高]
            color (str, optional): [真实框目标颜色]. Defaults to ''.
            tool (str, optional): [bbox工具]. Defaults to ''.
            difficult (int, optional): [困难样本]. Defaults to 0.
            distance (float, optional): [真实框中心点距离]. Defaults to 0.
            occlusion (float, optional): [真实框遮挡率]. Defaults to 0.
        """

        self.bbox_clss = bbox_clss
        self.bbox_xywh = xywh
        self.color = color
        self.tool = tool
        self.difficult = difficult
        self.distance = distance
        self.occlusion = occlusion


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
            self.area = int(cv2.contourArea(np.array(self.segmentation)))
        else:
            self.area = area
        self.iscrowd = int(iscrowd)


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


class OBJECT(BBOX, SEGMENTATION, KEYPOINTS):
    """标注物体类"""

    def __init__(self,
                 object_clss: str,
                 bbox_clss: str,
                 segmentation_clss: str,
                 keypoints_clss: str,

                 xywh: list,
                 segmentation: list,

                 num_keypoints: int,
                 keypoints: list,

                 bbox_color: str = '',
                 bbox_tool: str = '',
                 bbox_difficult: int = 0,
                 bbox_distance: float = 0,
                 bbox_occlusion: float = 0,

                 segmentation_area: int = None,
                 segmentation_iscrowd: int = 0,
                 ) -> None:
        """[summary]

        Args:
            object_clss (str): [description]
            bbox_clss (str): [description]
            segmentation_clss (str): [description]
            keypoints_clss (str): [description]
            xywh (list): [description]
            segmentation (list): [description]
            num_keypoints (int): [description]
            keypoints (list): [description]
            bbox_color (str, optional): [description]. Defaults to ''.
            bbox_tool (str, optional): [description]. Defaults to ''.
            bbox_difficult (int, optional): [description]. Defaults to 0.
            bbox_distance (float, optional): [description]. Defaults to 0.
            bbox_occlusion (float, optional): [description]. Defaults to 0.
            segmentation_area (int, optional): [description]. Defaults to None.
            segmentation_iscrowd (int, optional): [description]. Defaults to 0.
        """

        BBOX.__init__(self, bbox_clss, xywh,
                      color=bbox_color, tool=bbox_tool, difficult=bbox_difficult,
                      distance=bbox_distance, occlusion=bbox_occlusion)
        SEGMENTATION.__init__(self, segmentation_clss, segmentation,
                              area=segmentation_area, iscrowd=segmentation_iscrowd)
        KEYPOINTS.__init__(self, keypoints_clss, num_keypoints, keypoints)
        self.object_clss = object_clss

    def get_outer_bounding_box(self):
        """[将分割按最外围矩形框转换为bbox]
        """

        segmentation = np.asarray(self.segmentation)
        self.min_x = int(np.min(segmentation[:, 0]))
        self.min_y = int(np.min(segmentation[:, 1]))
        self.max_x = int(np.max(segmentation[:, 0]))
        self.max_y = int(np.max(segmentation[:, 1]))

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

    def true_object_list_updata(self, object_data: OBJECT) -> None:
        """[为per_image对象true_box_list成员添加元素]

        Args:
            true_object_data (TRUE_OBJECT): [TRUE_OBJECT类变量]
        """

        self.object_list.append(object_data)

    def modify_object_list(self, input_dataset) -> None:
        """[修改真实框类别]

        Args:
            class_modify_dict (dict): [类别修改字典]
        """
        for task in input_dataset.task:
            if task.class_modify_dict is not None:
                for one_object in self.object_list:
                    # 遍历融合类别文件字典，完成label中的类别修改，
                    # 若此bbox类别属于混合标签类别列表，则返回该标签在混合类别列表的索引值，修改类别。
                    for (key, value) in task.class_modify_dict.items():
                        if one_object.clss in set(value):
                            one_object.clss = key

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
