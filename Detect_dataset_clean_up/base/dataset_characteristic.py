'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-09 21:55:18
LastEditors: Leidi
LastEditTime: 2021-11-03 10:29:07
'''

# 输入数据集图片、annotation文件格式
dataset_file_form = {'pascal_voc': {'image': 'jpg',
                                    'annotation': 'xml'
                                    },
                     'yolo': {'image': 'jpg',
                              'annotation': 'txt'
                              },
                     'coco2017': {'image': 'jpg',
                                  'annotation': 'json'
                                  },
                     'tt100k': {'image': 'jpg',
                                'annotation': 'json'
                                },
                     'sjt': {'image': 'jpg',
                             'annotation': 'json'
                             },
                     'myxb': {'image': 'jpg',
                              'annotation': 'json'
                              },
                     'hy_highway': {'image': 'png',
                                    'annotation': 'xml'
                                    },
                     'kitti': {'image': 'png',
                               'annotation': 'txt'
                               },
                     'lisa': {'image': 'jpg',
                              'annotation': 'csv'
                              },
                     'cctsdb': {'image': 'png',
                                'annotation': 'txt'
                                },
                     'cvat_coco2017': {'image': 'jpg',
                                       'annotation': 'json'
                                       },
                     'huawei_segment': {'image': 'jpg',
                                        'annotation': 'json'
                                        }
                     }

# 数据集使用指定层级文件夹更名
ANNOTATAION_RENAME_WITH_FOLDER = {'lisa': -1}
IMAGE_RENAME_WITH_FOLDER = {}

temp_arch = {'image': 'source_images',
             'annotation': 'temp_annotations',
             'information': 'temp_informations'}

temp_form = {'image': 'jpg',
             'annotation': 'xml'}
