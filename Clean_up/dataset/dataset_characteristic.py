'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-09 21:55:18
LastEditors: Leidi
LastEditTime: 2022-01-10 10:39:21
'''
# 数据集任务类型列表
ONE_TASK = ['Detection', 'Instance_segmentation', 'Semantic_segmentation']
MULTI_TASK = ['Multi_task']

# 输入数据集图片、annotation文件格式
DATASET_FILE_FORM = {'apolloscape_lane_segment': {'image': 'jpg',
                                                  'annotation': 'png'
                                                  },
                     'bdd100k': {'image': 'jpg',
                                 'annotation': 'json'
                                 },
                     'cvat_image_1_1': {'image': 'jpg',
                                        'annotation': 'xml'
                                        },
                     'cityscapes_val': {'image': 'png',
                                        'annotation': 'json'
                                        },
                     'COCO2017': {'image': 'jpg',
                                  'annotation': 'json'
                                  },
                     'cityscapes': {'image': 'png',
                                    'annotation': 'json'
                                    },
                     'hy_val': {'image': 'png',
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

# 暂存数据集图片、annotation文件格式
TEMP_ARCH = {'image': 'source_images',
             'annotation': 'temp_annotations',
             'information': 'temp_informations'}
TEMP_FORM = {'image': 'png',
             'annotation': 'json'}
