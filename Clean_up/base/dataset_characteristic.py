'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-09 21:55:18
LastEditors: Leidi
LastEditTime: 2022-02-15 14:02:46
'''


# 输入数据集图片、annotation文件格式
TARGET_DATASET_FILE_FORM = {'CVAT_IMAGE_1_1': {'image': 'jpg',
                                               'annotation': 'xml'
                                               },
                            'CITYSCAPES_VAL': {'image': 'png',
                                               'annotation': 'json'
                                               },
                            'COCO2017': {'image': 'jpg',
                                         'annotation': 'json'
                                         },
                            'CITYSCAPES': {'image': 'png',
                                           'annotation': 'json'
                                           },
                            'YOLO': {'image': 'jpg',
                                     'annotation': 'txt'
                                     }
                            }
