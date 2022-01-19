'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-01-19 17:07:10
'''
import time
import shutil
from PIL import Image
import multiprocessing

import dataset
from utils.utils import *
from base.image_base import *
from utils import image_form_transform
from .dataset_characteristic import *
from dataset.dataset_base import Dataset_Base


class COCO2017(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)

    def source_dataset_copy_image_and_annotation(self):
        print('\nStart source dataset copy image and annotation:')
        print('Copy images: ')
        for root, _, files in tqdm(os.walk(self.dataset_input_folder)):
            pool = multiprocessing.Pool(self.workers)
            for n in tqdm(files):
                if n.endswith(self.source_dataset_image_form):
                    pool.apply_async(self.source_dataset_copy_image,
                                     args=(root, n,), error_callback=err_call_back)
            pool.close()
            pool.join()
            print('Move images count: {}\n'.format(
                len(os.listdir(self.source_dataset_images_folder))))

        print('Copy annotations: ')
        for root, _, files in tqdm(os.walk(self.dataset_input_folder)):
            pool = multiprocessing.Pool(self.workers)
            for n in tqdm(files):
                if n.endswith(self.source_dataset_annotation_form):
                    pool.apply_async(self.source_dataset_copy_annotation,
                                     args=(root, n,), error_callback=err_call_back)
            pool.close()
            pool.join()
        print('Move annotations count: {}\n'.format(
            len(os.listdir(self.source_dataset_annotations_folder))))

        return

    def transform_to_temp_dataset(self):
        print('\nStart transform to temp dataset:')
        success_count = 0
        fail_count = 0
        no_object = 0
        temp_file_name_list = []

        for source_annotation_name in tqdm(os.listdir(self.source_dataset_annotations_folder)):
            source_annotation_path = os.path.join(
                self.source_dataset_annotations_folder, source_annotation_name)
            with open(source_annotation_path, 'r') as f:
                data = json.loads(f.read())

            del f

            class_dict = {}
            for n in data['categories']:
                class_dict['%s' % n['id']] = n['name']

            # 获取data字典中images内的图片信息，file_name、height、width
            total_annotations_dict = multiprocessing.Manager().dict()
            pool = multiprocessing.Pool(self.workers)
            for image_base_information in tqdm(data['images']):
                pool.apply_async(func=self.load_image_base_information, args=(
                    image_base_information, total_annotations_dict,),
                    error_callback=err_call_back)
            pool.close()
            pool.join()

            # 读取目标标注信息
            total_image_annotation_list = []
            pool = multiprocessing.Pool(self.workers)
            for id, one_annotation in tqdm(enumerate(data['annotations'])):
                total_image_annotation_list.append(pool.apply_async(func=self.load_image_annotation, args=(
                    id, one_annotation, class_dict, total_annotations_dict,),
                    error_callback=err_call_back))
            pool.close()
            pool.join()

            del data

            total_images_data_dict = {}
            for image_true_annotation in total_image_annotation_list:
                if image_true_annotation.get() is None:
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
            process_temp_file_name_list = multiprocessing.Manager().list()
            process_output = multiprocessing.Manager().dict({'success_count': 0,
                                                             'fail_count': 0,
                                                             'no_object': 0,
                                                             'temp_file_name_list': process_temp_file_name_list
                                                             })
            pool = multiprocessing.Pool(self.workers)
            for _, image in tqdm(total_images_data_dict.items()):
                pool.apply_async(func=self.output_temp_annotation, args=(
                    image, process_output,),
                    error_callback=err_call_back)
            pool.close()
            pool.join()

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

        return

    def source_dataset_copy_image(self, root: str, n: str) -> None:
        """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀]

        Args:
            dataset (dict): [数据集信息字典]
            root (str): [文件所在目录]
            n (str): [文件名]

        """

        image = os.path.join(root, n)
        temp_image = os.path.join(
            self.source_dataset_images_folder, self.file_prefix + n)
        image_suffix = os.path.splitext(n)[-1].replace('.', '')
        if image_suffix != self.target_dataset_image_form:
            image_transform_type = image_suffix + \
                '_' + self.target_dataset_image_form
            image_form_transform.__dict__[
                image_transform_type](image, temp_image)
            return
        else:
            shutil.copy(image, temp_image)
            return

    def source_dataset_copy_annotation(self, root: str, n: str) -> None:
        """[复制源数据集标签文件至目标数据集中的source_annotations中]

        Args:
            dataset (dict): [数据集信息字典]
            root (str): [文件所在目录]
            n (str): [文件名]
        """

        annotation = os.path.join(root, n)
        temp_annotation = os.path.join(
            self.source_dataset_annotations_folder, n)
        shutil.copy(annotation, temp_annotation)

        return

    def load_image_base_information(self, image_base_information: dict, total_annotations_dict: dict) -> None:
        """[读取标签获取图片基础信息，并添加至each_annotation_images_data_dict]

        Args:
            dataset (dict): [数据集信息字典]
            one_image_base_information (dict): [单个数据字典信息]
            each_annotation_images_data_dict进程间通信字典 (dict): [each_annotation_images_data_dict进程间通信字典]
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
        """[读取单个标签详细信息，并添加至each_annotation_images_data_dict]

        Args:
            id(int): [标注id]
            dataset (dict): [数据集信息字典]
            one_annotation (dict): [单个数据字典信息]
            class_dict (dict): [类别字典]
            process_output (dict): [each_annotation_images_data_dict进程间通信字典]

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

        cls = class_dict[str(one_annotation['category_id'])]     # 获取object类别
        cls = cls.replace(' ', '').lower()
        for _, task_class_dict in self.task_dict.items():
            if cls not in task_class_dict['Source_dataset_class']:
                return
        image = each_annotation_images_data_dict[ann_image_id]

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

        one_object = OBJECT(id, cls, cls, cls, cls,
                            box_xywh, segmentation, keypoints_num, keypoints,
                            segmentation_area=segmentation_area,
                            segmentation_iscrowd=segmentation_iscrowd
                            )

        return ann_image_id, one_object

    def output_temp_annotation(self, image: IMAGE, process_output: dict) -> None:
        """[输出单个标签详细信息至temp annotation]

        Args:
            dataset (dict): [数据集信息字典]
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
    def target_dataset(dataset_instance: object):
        """[输出temp dataset annotation]

        Args:
            dataset (Dataset): [dataset]
        """

        print('\nStart transform to target dataset:')
        for dataset_temp_annotation_path_list in tqdm(dataset_instance.temp_divide_file_list[1:-1]):
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
            class_count = 0
            for task, task_class_dict in dataset_instance.task_dict.items():
                for n, cls in enumerate(task_class_dict['Target_dataset_class']):
                    category_item = {'supercategory': 'none',
                                     'id': n + class_count,
                                     'name': cls}
                    coco['categories'].append(category_item)
                    class_count += len(task_class_dict['Target_dataset_class'])

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
                pool = multiprocessing.Pool(dataset_instance.workers)
                for n, temp_annotation_path in tqdm(enumerate(annotation_path_list)):
                    image_information_list.append(
                        pool.apply_async(func=dataset.__dict__[dataset_instance.target_dataset_style].get_image_information,
                                         args=(dataset_instance, coco,
                                               n, temp_annotation_path,),
                                         error_callback=err_call_back))
                pool.close()
                pool.join()

                for n in tqdm(image_information_list):
                    coco['images'].append(n.get())
                del image_information_list

                # 读取标签标注基础信息
                print('Start load annotation:')
                annotations_list = []
                pool = multiprocessing.Pool(dataset_instance.workers)
                for n, temp_annotation_path in tqdm(enumerate(annotation_path_list)):
                    annotations_list.append(
                        pool.apply_async(func=dataset.__dict__[dataset_instance.target_dataset_style].get_annotation,
                                         args=(dataset_instance, n,
                                               temp_annotation_path,
                                               task_class_dict['Target_dataset_class'],
                                               class_count,),
                                         error_callback=err_call_back))
                pool.close()
                pool.join()

                annotation_id = 0
                for n in tqdm(annotations_list):
                    for m in n.get():
                        m['id'] = annotation_id
                        coco['annotations'].append(m)
                        annotation_id += 1
                del annotations_list

            json.dump(coco, open(annotation_output_path, 'w'))

        return

    @staticmethod
    def get_image_information(dataset_instance: object, coco: dict, n: int, temp_annotation_path: str) -> None:
        """[读取暂存annotation]

        Args:
            dataset_instance (): [数据集信息字典]
            temp_annotation_path (str): [annotation路径]

        Returns:
            IMAGE: [输出IMAGE类变量]
        """

        with open(temp_annotation_path, 'r') as f:
            image_name = temp_annotation_path.split(
                os.sep)[-1].replace('.json', '.' + dataset_instance.temp_image_form)
            image_path = os.path.join(
                dataset_instance.temp_images_folder, image_name)
            if os.path.splitext(image_path)[-1] == 'png':
                img = Image.open(image_path)
                height, width = img.height, img.width
                channels = 3
            else:
                image_size = cv2.imread(image_path).shape
                height = int(image_size[0])
                width = int(image_size[1])
                channels = int(image_size[2])

            image = IMAGE(image_name, image_name,
                          image_path, height, width, channels, [])

        return image

    @staticmethod
    def get_annotation(dataset_instance: object, n: int,
                       temp_annotation_path: str,
                       task_class: list,
                       task_class_count: int) -> None:
        """[获取暂存标注信息]

        Args:
            dataset (dict): [数据集信息字典]
            n (int): [图片id]
            temp_annotation_path (str): [暂存标签路径]
        """

        image = dataset_instance.TEMP_LOAD(
            dataset_instance, temp_annotation_path)
        if image == None:
            return
        # 获取图片分割信息
        one_image_annotations_list = []
        for object in image.object_list:
            segmentation = np.asarray(
                object.segmentation).flatten().tolist()
            one_image_annotations_list.append({'bbox': object.box_xywh,
                                               'segmentation': [segmentation],
                                               'area': object.segmentation_area,
                                               'iscrowd': object.iscrowd,
                                               'keypoints':object.keypoints,
                                               'num_keypoints': object.keypoints_num,
                                               'image_id': n,
                                               'category_id': task_class.index(object.clss)
                                               + task_class_count,
                                               'id': 0})

        return one_image_annotations_list

    @staticmethod
    def target_dataset_folder(dataset_instance):
        print('\nStart build target dataset folder:')
