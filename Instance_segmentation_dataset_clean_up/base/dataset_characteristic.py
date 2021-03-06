'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-09 21:55:18
LastEditors: Leidi
LastEditTime: 2021-12-15 17:15:48
'''


# 输入数据集图片、annotation文件格式
dataset_file_form = {'coco2017': {'image': 'jpg',
                                  'annotation': 'json'
                                  },
                     'cityscapes': {'image': 'png',
                                    'annotation': 'json'
                                    },
                     'pascal_voc': {'image': 'jpg',
                                    'annotation': 'xml'
                                    },
                     'bdd100k': {'image': 'jpg',
                                 'annotation': 'json'
                                 },
                     'apolloscape_lane_segment': {'image': 'jpg',
                                                  'annotation': 'png'
                                                  },
                     'huaweiyun_segment': {'image': 'jpg',
                                           'annotation': 'json'
                                           },
                     'cvat_image_1_1': {'image': 'jpg',
                                        'annotation': 'xml'
                                        },
                     'yunce_segment_coco': {'image': 'jpg',
                                            'annotation': 'json'
                                            },
                     'yunce_segment_coco_one_image': {'image': 'jpg',
                                                      'annotation': 'json'
                                                      },
                     'hy_val': {'image': 'png',
                                'annotation': 'json'
                                },
                     'cityscapes_val': {'image': 'png',
                                        'annotation': 'json'
                                        }
                     }

# 暂存数据集图片、annotation文件格式
temp_arch = {'image': 'source_images',
             'annotation': 'temp_annotations',
             'information': 'temp_informations'}
temp_form = {'image': 'png',
             'annotation': 'json'}
