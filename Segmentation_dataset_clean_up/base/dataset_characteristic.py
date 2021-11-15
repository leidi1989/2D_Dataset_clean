'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-09 21:55:18
LastEditors: Leidi
LastEditTime: 2021-11-15 13:43:05
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
                     'huawei_segment': {'image': 'jpg',
                                        'annotation': 'json'
                                        },
                     'cvat_image_1_1': {'image': 'jpg',
                                        'annotation': 'xml'
                                        },
                     'huawei_segment': {'image': 'jpg',
                                        'annotation': 'json'
                                        },
                     'hy_val': {'image': 'jpg',
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
