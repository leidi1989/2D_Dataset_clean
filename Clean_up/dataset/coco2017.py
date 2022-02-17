'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-18 02:40:51
'''
import time
import shutil
from PIL import Image
import multiprocessing

from sqlalchemy import desc, false

import dataset
from utils.utils import *
from base.image_base import *
from base.dataset_base import Dataset_Base


class COCO2017(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg']
        self.source_dataset_annotation_form = 'json'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count()

    def transform_to_temp_dataset(self) -> None:
        """[转换标注文件为暂存标注]
        """

        print('\nStart transform to temp dataset:')
        success_count = 0
        fail_count = 0
        no_object = 0
        temp_file_name_list = []

        for source_annotation_name in tqdm(os.listdir(self.source_dataset_annotations_folder),
                                           desc='Total annotations'):
            source_annotation_path = os.path.join(
                self.source_dataset_annotations_folder, source_annotation_name)
            with open(source_annotation_path, 'r') as f:
                data = json.loads(f.read())
            del f

            class_dict = {}
            for n in data['categories']:
                class_dict['%s' % n['id']] = n['name']

            # 获取data字典中images内的图片信息，file_name、height、width
            pbar, update = multiprocessing_list_tqdm(
                data['images'], desc='Load image base information', leave=False)
            total_annotations_dict = multiprocessing.Manager().dict()
            pool = multiprocessing.Pool(self.workers)
            for image_base_information in data['images']:
                pool.apply_async(func=self.load_image_base_information,
                                 args=(image_base_information,
                                       total_annotations_dict,),
                                 callback=update,
                                 error_callback=err_call_back)
            pool.close()
            pool.join()
            pbar.close()

            # 读取目标标注信息
            pbar, update = multiprocessing_list_tqdm(
                data['annotations'], desc='Load image annotation', leave=False)
            total_image_annotation_list = []
            pool = multiprocessing.Pool(self.workers)
            for id, one_annotation in enumerate(data['annotations']):
                total_image_annotation_list.append(pool.apply_async(func=self.load_image_annotation,
                                                                    args=(
                                                                        id, one_annotation, class_dict, total_annotations_dict),
                                                                    callback=update,
                                                                    error_callback=err_call_back))
            pool.close()
            pool.join()
            pbar.close()

            del data

            total_images_data_dict = {}
            for image_true_annotation in total_image_annotation_list:
                if image_true_annotation.get()[1] is None:
                    continue
                if image_true_annotation.get()[0] not in total_images_data_dict:
                    total_images_data_dict[image_true_annotation.get(
                    )[0]] = total_annotations_dict[image_true_annotation.get()[0]]
                    total_images_data_dict[image_true_annotation.get()[0]].object_list.append(
                        image_true_annotation.get()[1])
                else:
                    total_images_data_dict[image_true_annotation.get()[0]].object_list.append(
                        image_true_annotation.get()[1])

            del total_annotations_dict, total_image_annotation_list

            # 输出读取的source annotation至temp annotation
            pbar, update = multiprocessing_list_tqdm(
                total_images_data_dict, desc='Output temp annotation', leave=False)
            process_temp_file_name_list = multiprocessing.Manager().list()
            process_output = multiprocessing.Manager().dict({'success_count': 0,
                                                             'fail_count': 0,
                                                             'no_object': 0,
                                                             'temp_file_name_list': process_temp_file_name_list
                                                             })
            pool = multiprocessing.Pool(self.workers)
            for _, image in total_images_data_dict.items():
                pool.apply_async(func=self.output_temp_annotation,
                                 args=(image, process_output,),
                                 callback=update,
                                 error_callback=err_call_back)
            pool.close()
            pool.join()
            pbar.close()

            # 更新输出统计
            success_count += process_output['success_count']
            fail_count += process_output['fail_count']
            no_object += process_output['no_object']
            temp_file_name_list += process_output['temp_file_name_list']

        # 输出读取统计结果
        print('\nSource dataset convert to temp dataset file count: ')
        print('Total annotations:         \t {} '.format(
            len(os.listdir(self.source_dataset_annotations_folder))))
        print('Convert fail:              \t {} '.format(fail_count))
        print('No object delete images: \t {} '.format(no_object))
        print('Convert success:           \t {} '.format(success_count))
        self.temp_annotation_name_list = temp_file_name_list
        print('Source dataset annotation transform to temp dataset end.')

        return

    def load_image_base_information(self, image_base_information: dict, total_annotations_dict: dict) -> None:
        """读取标签获取图片基础信息, 并添加至each_annotation_images_data_dict

        Args:
            image_base_information (dict): 图片基础信息字典
            total_annotations_dict (dict): 全部标注信息字典
        """

        image_id = image_base_information['id']
        image_name = os.path.splitext(image_base_information['file_name'])[
            0] + '.' + self.temp_image_form
        image_name_new = self.file_prefix + image_name
        image_path = os.path.join(
            self.temp_images_folder, image_name_new)
        img = Image.open(image_path)
        height, width = img.height, img.width
        channels = 3
        # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
        # 并将初始化后的对象存入total_images_data_list
        image = IMAGE(image_name, image_name_new,
                      image_path, height, width, channels, [])
        total_annotations_dict.update({image_id: image})

        return

    def load_image_annotation(self, id: int, one_annotation: dict,
                              class_dict: dict, each_annotation_images_data_dict: dict) -> list:
        """[读取单个标签详细信息, 并添加至each_annotation_images_data_dict]

        Args:
            id(int): [标注id]
            one_annotation (dict): [单个数据字典信息]
            class_dict (dict): [类别字典]
            each_annotation_images_data_dict (dict): [进程间通信字典]

        Returns:
            list: [ann_image_id, true_box_list, true_segmentation_list]
        """

        box_xywh = []
        segmentation = []
        segmentation_area = None
        segmentation_iscrowd = 0
        keypoints_num = 0
        keypoints = []

        ann_image_id = one_annotation['image_id']   # 获取此object图片id

        clss = class_dict[str(one_annotation['category_id'])]     # 获取object类别
        clss = clss.replace(' ', '').lower()
        if clss not in self.total_task_source_class_list:
            return ann_image_id, None
        if ann_image_id in each_annotation_images_data_dict.keys():
            image = each_annotation_images_data_dict[ann_image_id]
        else:
            return ann_image_id, None

        # 获取真实框信息
        if 'bbox' in one_annotation and len(one_annotation['bbox']):
            box = [one_annotation['bbox'][0],
                   one_annotation['bbox'][1],
                   one_annotation['bbox'][0] + one_annotation['bbox'][2],
                   one_annotation['bbox'][1] + one_annotation['bbox'][3]]
            xmin = max(min(int(box[0]), int(box[2]),
                           int(image.width)), 0.)
            ymin = max(min(int(box[1]), int(box[3]),
                           int(image.height)), 0.)
            xmax = min(max(int(box[2]), int(box[0]), 0.),
                       int(image.width))
            ymax = min(max(int(box[3]), int(box[1]), 0.),
                       int(image.height))
            box_xywh = [xmin, ymin, xmax-xmin, ymax-ymin]

        # 获取真实语义分割信息
        if 'segmentation' in one_annotation and len(one_annotation['segmentation']):
            for one_seg in one_annotation['segmentation']:
                segment = []
                point = []
                for i, x in enumerate(one_seg):
                    if 0 == i % 2:
                        point.append(x)
                    else:
                        point.append(x)
                        point = list(map(int, point))
                        segment.append(point)
                        if 2 != len(point):
                            print('Segmentation label wrong: ',
                                  each_annotation_images_data_dict[ann_image_id].image_name_new)
                            continue
                        point = []
                segmentation = segment
                segmentation_area = one_annotation['area']
                if '1' == one_annotation['iscrowd']:
                    segmentation_iscrowd = 1

        # 关键点信息
        if 'keypoints' in one_annotation and len(one_annotation['keypoints']) \
                and 'num_keypoints' in one_annotation:
            keypoints_num = one_annotation['num_keypoints']
            keypoints = one_annotation['keypoints']

        one_object = OBJECT(id,
                            clss,
                            box_clss=clss,
                            segmentation_clss=clss,
                            keypoints_clss=clss,
                            box_xywh=box_xywh,
                            segmentation=segmentation,
                            keypoints_num=keypoints_num,
                            keypoints=keypoints,
                            need_convert=self.task_convert,
                            segmentation_area=segmentation_area,
                            segmentation_iscrowd=segmentation_iscrowd,
                            )

        return ann_image_id, one_object

    def output_temp_annotation(self, image: IMAGE, process_output: dict) -> None:
        """[输出单个标签详细信息至temp annotation]

        Args:
            image (IMAGE): [IMAGE类实例]
            process_output (dict): [进程间计数通信字典]
        """

        if image == None:
            return

        temp_annotation_output_path = os.path.join(
            self.temp_annotations_folder,
            image.file_name_new + '.' + self.temp_annotation_form)
        image.modify_object_list(self)
        image.object_pixel_limit(self)
        if 0 == len(image.object_list):
            print('{} no object, has been delete.'.format(
                image.image_name_new))
            os.remove(image.image_path)
            process_output['no_object'] += 1
            process_output['fail_count'] += 1
            return
        if image.output_temp_annotation(temp_annotation_output_path):
            process_output['temp_file_name_list'].append(image.file_name_new)
            process_output['success_count'] += 1
        else:
            print('errow output temp annotation: {}'.format(image.file_name_new))
            process_output['fail_count'] += 1

        return

    @staticmethod
    def target_dataset(dataset_instance: Dataset_Base) -> None:
        """[输出target annotation]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]
        """

        print('\nStart transform to target dataset:')
        for dataset_temp_annotation_path_list in tqdm(dataset_instance.temp_divide_file_list[1:-1],
                                                      desc='Transform to target dataset'):
            # 声明coco字典及基础信息
            coco = {'info': {'description': 'COCO 2017 Dataset',
                             'url': 'http://cocodataset.org',
                             'version': '1.0',
                             'year': 2017,
                             'contributor': 'leidi',
                             'date_created': time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
                             },
                    'licenses': [
                {
                    'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
                    'id': 1,
                    'name': 'Attribution-NonCommercial-ShareAlike License'
                },
                {
                    'url': 'http://creativecommons.org/licenses/by-nc/2.0/',
                    'id': 2,
                    'name': 'Attribution-NonCommercial License'
                },
                {
                    'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/',
                    'id': 3,
                    'name': 'Attribution-NonCommercial-NoDerivs License'
                },
                {
                    'url': 'http://creativecommons.org/licenses/by/2.0/',
                    'id': 4,
                    'name': 'Attribution License'
                },
                {
                    'url': 'http://creativecommons.org/licenses/by-sa/2.0/',
                    'id': 5,
                    'name': 'Attribution-ShareAlike License'
                },
                {
                    'url': 'http://creativecommons.org/licenses/by-nd/2.0/',
                    'id': 6,
                    'name': 'Attribution-NoDerivs License'
                },
                {
                    'url': 'http://flickr.com/commons/usage/',
                    'id': 7,
                    'name': 'No known copyright restrictions'
                },
                {
                    'url': 'http://www.usa.gov/copyright.shtml',
                    'id': 8,
                    'name': 'United States Government Work'
                }
            ],
                'images': [],
                'annotations': [],
                'categories': []
            }

            # 将class_list_new转换为coco格式字典
            for n, clss in enumerate(dataset_instance
                                     .temp_merge_class_list['Merge_target_dataset_class_list']):
                category_item = {'supercategory': 'none',
                                 'id': n,
                                 'name': clss}
                coco['categories'].append(category_item)

            annotation_output_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder, 'instances_' + os.path.splitext(
                    dataset_temp_annotation_path_list.split(os.sep)[-1])[0]
                + str(2017) + '.' + dataset_instance.target_dataset_annotation_form)
            annotation_path_list = []
            with open(dataset_temp_annotation_path_list, 'r') as f:
                for n in f.readlines():
                    annotation_path_list.append(n.replace('\n', '')
                                                .replace(dataset_instance.source_dataset_images_folder,
                                                         dataset_instance.temp_annotations_folder)
                                                .replace(dataset_instance.target_dataset_image_form,
                                                         dataset_instance.temp_annotation_form))

            # 读取标签图片基础信息
            print('Start load image information:')
            image_information_list = []
            pbar, update = multiprocessing_list_tqdm(
                annotation_path_list, desc='Load image information')
            pool = multiprocessing.Pool(dataset_instance.workers)
            for n, temp_annotation_path in enumerate(annotation_path_list):
                image_information_list.append(
                    pool.apply_async(func=dataset.__dict__[dataset_instance.target_dataset_style].get_image_information,
                                     args=(dataset_instance, coco,
                                           n, temp_annotation_path,),
                                     callback=update,
                                     error_callback=err_call_back))
            pool.close()
            pool.join()
            pbar.close()

            for n in image_information_list:
                coco['images'].append(n.get())
            del image_information_list

            # 读取图片标注基础信息
            print('Start load annotation:')
            for task, task_class_dict in tqdm(dataset_instance.task_dict.items(), desc='Load each task annotation'):
                if task_class_dict is None:
                    continue
                annotations_list = []
                pbar, update = multiprocessing_list_tqdm(
                    annotation_path_list, desc='Load annotation', leave=False)
                pool = multiprocessing.Pool(dataset_instance.workers)
                for n, temp_annotation_path in tqdm(enumerate(annotation_path_list)):
                    annotations_list.append(
                        pool.apply_async(func=dataset.__dict__[dataset_instance.target_dataset_style].get_annotation,
                                         args=(dataset_instance, n,
                                               temp_annotation_path,
                                               task,
                                               task_class_dict,),
                                         callback=update,
                                         error_callback=err_call_back))
                pool.close()
                pool.join()
                pbar.close()

                annotation_id = 0
                for n in tqdm(annotations_list):
                    for m in n.get():
                        m['id'] = annotation_id
                        coco['annotations'].append(m)
                        annotation_id += 1
                del annotations_list

            print('Output image annotation to json.')
            json.dump(coco, open(annotation_output_path, 'w'))

        return

    @staticmethod
    def get_image_information(dataset_instance: Dataset_Base, coco: dict, n: int, temp_annotation_path: str) -> dict:
        """读取暂存annotation

        Args:
            dataset_instance (object): 数据集信息字典
            coco (dict): coco格式基础信息
            n (int): 图片id
            temp_annotation_path (str): 暂存标注路径

        Returns:
            dict: 图片基础信息
        """

        image = dataset_instance.TEMP_LOAD(
            dataset_instance, temp_annotation_path)
        if image == None:
            return
        # 图片基础信息
        image_information = {'license': random.randint(0, len(coco['licenses'])),
                             'file_name': image.image_name_new,
                             'coco_url': 'None',
                             'height': image.height,
                             'width': image.width,
                             'date_captured': time.strftime('%Y/%m/%d %H:%M:%S', time.localtime()),
                             'flickr_url': 'None',
                             'id': n
                             }

        return image_information

    @staticmethod
    def get_annotation(dataset_instance: Dataset_Base,
                       n: int,
                       temp_annotation_path: str,
                       task: str,
                       task_class_dict: dict) -> list:
        """获取暂存标注信息

        Args:
            dataset_instance (object): 数据集信息字典
            n (int): 图片id
            temp_annotation_path (str): 暂存标签路径
            task (str): 任务类型
            task_class_dict (dict): 任务对应类别字典

        Returns:
            list: 图片标注信息字典列表
        """

        image = dataset_instance.TEMP_LOAD(
            dataset_instance, temp_annotation_path)
        if image == None:
            return
        # 获取图片标注信息
        one_image_annotations_list = []
        for object in image.object_list:
            # class
            if task == 'Detection':
                clss = object.box_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                category_id = dataset_instance.temp_merge_class_list['Merge_target_dataset_class_list'].index(
                    clss)
                one_image_annotations_list.append({'bbox': object.box_xywh,
                                                   'area': object.box_get_area(),
                                                   'iscrowd': object.segmentation_iscrowd,
                                                   'keypoints': object.keypoints,
                                                   'num_keypoints': object.keypoints_num,
                                                   'image_id': n,
                                                   'category_id': category_id,
                                                   'id': 0})
            elif task == 'Semantic_segmentation':
                segmentation = np.asarray(
                    object.segmentation).flatten().tolist()
                clss = object.segmentation_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                category_id = dataset_instance.temp_merge_class_list['Merge_target_dataset_class_list'].index(
                    clss)
                one_image_annotations_list.append({'segmentation': [segmentation],
                                                   'area': object.segmentation_area,
                                                   'iscrowd': object.segmentation_iscrowd,
                                                   'keypoints': object.keypoints,
                                                   'num_keypoints': object.keypoints_num,
                                                   'image_id': n,
                                                   'category_id': category_id,
                                                   'id': 0})
            elif task == 'Instance_segmentation':
                segmentation = np.asarray(
                    object.segmentation).flatten().tolist()
                clss = object.segmentation_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                category_id = dataset_instance.temp_merge_class_list['Merge_target_dataset_class_list'].index(
                    clss)
                one_image_annotations_list.append({'bbox': object.box_xywh,
                                                   'segmentation': [segmentation],
                                                   'area': object.segmentation_area,
                                                   'iscrowd': object.segmentation_iscrowd,
                                                   'keypoints': object.keypoints,
                                                   'num_keypoints': object.keypoints_num,
                                                   'image_id': n,
                                                   'category_id': category_id,
                                                   'id': 0})
            elif task == 'Keypoint':
                segmentation = np.asarray(
                    object.segmentation).flatten().tolist()
                clss = object.keypoints_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                one_image_annotations_list.append({'bbox': object.box_xywh,
                                                   'segmentation': [segmentation],
                                                   'area': object.segmentation_area,
                                                   'iscrowd': object.segmentation_iscrowd,
                                                   'keypoints': object.keypoints,
                                                   'num_keypoints': object.keypoints_num,
                                                   'image_id': n,
                                                   'category_id': dataset_instance
                                                   .temp_merge_class_list['Merge_target_dataset_class_list']
                                                   .index(clss),
                                                   'id': 0})

        return one_image_annotations_list

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取COCO2017数据集图片类检测列表]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []
        dataset_instance.total_file_name_path = total_file(
            dataset_instance.temp_informations_folder)
        dataset_instance.target_check_file_name_list = os.listdir(
            dataset_instance.target_dataset_annotations_folder)  # 读取target_annotations_folder文件夹下的全部文件名
        images_data_list = []
        images_data_dict = {}
        print('Start load target annotations:')
        for target_annotation in tqdm(dataset_instance.target_check_file_name_list,
                                      desc='Loading instances_train2017.json'):
            if target_annotation != 'instances_train2017.json':
                continue
            target_annotation_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder, target_annotation)
            with open(target_annotation_path, 'r') as f:
                data = json.loads(f.read())
            name_dict = {}
            for one_name in data['categories']:
                name_dict['%s' % one_name['id']] = one_name['name']

            total_image_count = 0
            for d in tqdm(data['images'],
                          desc='Count images',
                          leave=False):
                total_image_count += 1
            check_images_count = min(
                dataset_instance.target_dataset_annotations_check_count, total_image_count)
            check_image_id_list = [random.randint(
                0, total_image_count)for i in range(check_images_count)]

            for n in tqdm(check_image_id_list,
                          desc='Load image base information',
                          leave=False):
                d = data['images'][n]
                img_id = d['id']
                img_name = d['file_name']
                img_name_new = img_name
                img_path = os.path.join(
                    dataset_instance.temp_images_folder, img_name_new)
                img = Image.open(img_path)
                height, width = img.height, img.width
                channels = 3
                # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
                # 并将初始化后的对象存入total_images_data_list
                image = IMAGE(img_name, img_name_new,
                              img_path, height, width, channels, [])
                images_data_dict.update({img_id: image})

            for one_annotation in tqdm(data['annotations'],
                                       desc='Load object information',
                                       leave=False):
                if one_annotation['image_id'] in images_data_dict:
                    ann_image_id = one_annotation['image_id']   # 获取此bbox图片id
                    box_xywh = []
                    segmentation = []
                    segmentation_area = None
                    segmentation_iscrowd = 0
                    keypoints_num = 0
                    keypoints = []
                    clss = name_dict[str(one_annotation['category_id'])]
                    clss = clss.replace(' ', '').lower()
                    image = images_data_dict[ann_image_id]

                    # 获取真实框信息
                    if 'bbox' in one_annotation and len(one_annotation['bbox']):
                        box = [one_annotation['bbox'][0],
                               one_annotation['bbox'][1],
                               one_annotation['bbox'][0] +
                               one_annotation['bbox'][2],
                               one_annotation['bbox'][1] + one_annotation['bbox'][3]]
                        xmin = max(min(int(box[0]), int(box[2]),
                                       int(image.width)), 0.)
                        ymin = max(min(int(box[1]), int(box[3]),
                                       int(image.height)), 0.)
                        xmax = min(max(int(box[2]), int(box[0]), 0.),
                                   int(image.width))
                        ymax = min(max(int(box[3]), int(box[1]), 0.),
                                   int(image.height))
                        box_xywh = [xmin, ymin, xmax-xmin, ymax-ymin]

                    # 获取真实语义分割信息
                    if 'segmentation' in one_annotation and len(one_annotation['segmentation']):
                        for one_seg in one_annotation['segmentation']:
                            segment = []
                            point = []
                            for i, x in enumerate(one_seg):
                                if 0 == i % 2:
                                    point.append(x)
                                else:
                                    point.append(x)
                                    point = list(map(int, point))
                                    segment.append(point)
                                    if 2 != len(point):
                                        print('Segmentation label wrong: ',
                                              images_data_dict[ann_image_id].image_name_new)
                                        continue
                                    point = []
                            segmentation = segment
                            segmentation_area = one_annotation['area']
                            if '1' == one_annotation['iscrowd']:
                                segmentation_iscrowd = 1

                    # 关键点信息
                    if 'keypoints' in one_annotation and len(one_annotation['keypoints']) \
                            and 'num_keypoints' in one_annotation:
                        keypoints_num = one_annotation['num_keypoints']
                        keypoints = one_annotation['keypoints']

                    one_object = OBJECT(id, clss, clss, clss, clss,
                                        box_xywh, segmentation, keypoints_num, keypoints,
                                        dataset_instance.need_convert,
                                        segmentation_area=segmentation_area,
                                        segmentation_iscrowd=segmentation_iscrowd
                                        )
                    images_data_dict[ann_image_id].object_list.append(
                        one_object)

        for _, n in tqdm(images_data_dict.items(),
                         desc='Get result',
                         leave=False):
            images_data_list.append(n)
        random.shuffle(images_data_list)
        check_images_count = min(
            dataset_instance.target_dataset_annotations_check_count, len(images_data_list))
        check_images_list = images_data_list[0:check_images_count]

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成COCO 2017组织格式的数据集]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]
        """

        print('\nStart build target dataset folder:')
        output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder, 'coco2017'))
        shutil.rmtree(output_root)
        output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder, 'coco2017'))
        annotations_output_folder = check_output_path(
            os.path.join(output_root, 'annotations'))
        print('Start copy images:')
        for temp_divide_file in dataset_instance.temp_divide_file_list[1:4]:
            image_list = []
            coco_images_folder = os.path.splitext(
                temp_divide_file.split(os.sep)[-1])[0]
            image_output_folder = check_output_path(
                os.path.join(output_root, coco_images_folder + '2017'))
            with open(temp_divide_file, 'r') as f:
                for n in f.readlines():
                    image_list.append(n.replace('\n', ''))
            pbar, update = multiprocessing_list_tqdm(
                image_list, desc='Copy images', leave=False)
            pool = multiprocessing.Pool(dataset_instance.workers)
            for image_input_path in image_list:
                image_output_path = image_input_path.replace(
                    dataset_instance.temp_images_folder, image_output_folder)
                pool.apply_async(func=shutil.copy,
                                 args=(image_input_path, image_output_path,),
                                 callback=update,
                                 error_callback=err_call_back)
            pool.close()
            pool.join()
            pbar.close()

        print('Start copy annotations:')
        for root, dirs, files in os.walk(dataset_instance.target_dataset_annotations_folder):
            for n in tqdm(files, desc='Copy annotations'):
                annotations_input_path = os.path.join(root, n)
                annotations_output_path = annotations_input_path.replace(
                    dataset_instance.target_dataset_annotations_folder,
                    annotations_output_folder)
                shutil.copy(annotations_input_path, annotations_output_path)
        return
