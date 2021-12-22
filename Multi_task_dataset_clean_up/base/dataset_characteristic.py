'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-09 21:55:18
LastEditors: Leidi
LastEditTime: 2021-12-22 13:46:18
'''

# 输入数据集图片、annotation文件格式
dataset_file_form = {'huaweiyun_segment': {'image': 'jpg',
                                        'annotation': 'json'
                                        },
                     'yunce_segment_coco': {'image': 'jpg',
                                            'annotation': 'json'
                                            },
                     'yunce_segment_coco_one_image': {'image': 'jpg',
                                                      'annotation': 'json'
                                                      },
                     'bdd100k': {'image': 'jpg',
                                 'annotation': 'json',
                                 'detect_annotation': 'json',
                                 'segment_annotation': 'png'
                                 },
                     'yolop': {'image': 'jpg',
                               'annotation': 'json',
                               'detect_annotation': 'json',
                               'segment_annotation': 'png'
                               }
                     }

# 暂存数据集图片、annotation文件格式
temp_arch = {'image': 'source_images',
             'annotation': 'temp_annotations',
             'information': 'temp_informations'}
temp_form = {'image': 'jpg',
             'annotation': 'json',
             }
