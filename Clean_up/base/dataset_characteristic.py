'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-09 21:55:18
LastEditors: Leidi
LastEditTime: 2022-02-13 20:57:03
'''
# 输入数据集图片、annotation文件格式
TARGET_DATASET_FILE_FORM = {
    'apolloscape_lane_segment': {'image': 'jpg',
                                 'annotation': 'png'
                                 },
    'BDD100K': {'image': 'jpg',
                'annotation': 'json'
                },
    'cvat_image_1_1': {'image': 'jpg',
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
    'HY_VAL': {'image': 'png',
               'annotation': 'json'
               },
    'huaweiyun_segment': {'image': 'jpg',
                          'annotation': 'json'
                          },
    'hy_highway': {'image': 'png',
                   'annotation': 'xml'
                   },
    'pascal_voc': {'image': 'jpg',
                   'annotation': 'xml'
                   },
    'sjt': {'image': 'jpg',
            'annotation': 'json'
            },
    'tt100k': {'image': 'jpg',
               'annotation': 'json'
               },
    'yolo': {'image': 'jpg',
             'annotation': 'txt'
             },
    'yunce_segment_coco': {'image': 'jpg',
                           'annotation': 'json'
                           },
    'yunce_segment_coco_one_image': {'image': 'jpg',
                                     'annotation': 'json'
                                     },
}
