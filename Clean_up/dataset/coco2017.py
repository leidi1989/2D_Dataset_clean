'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-01-10 18:53:48
'''
import shutil
from PIL import Image
import multiprocessing

from utils.utils import *
from base.image_base import *
from utils import image_form_transform
from .dataset_characteristic import *
from dataset.dataset_base import Dataset_Base
from annotation.annotation_temp import TEMP_OUTPUT
from utils.convertion_function import true_segmentation_to_true_box
from utils.modify_class import modify_true_segmentation_list, modify_true_box_list


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
        no_segmentation = 0
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
                pool.apply_async(func=self.load_image_base_information(), args=(
                    self, image_base_information, total_annotations_dict,),
                    error_callback=err_call_back)
            pool.close()
            pool.join()

            # 读取目标标注信息
            total_image_segment_list = []
            pool = multiprocessing.Pool(self.workers)
            for one_annotation in tqdm(data['annotations']):
                total_image_segment_list.append(pool.apply_async(func=self.load_image_annotation(), args=(
                    self, one_annotation, class_dict, total_annotations_dict,),
                    error_callback=err_call_back))
            pool.close()
            pool.join()

            del data

            total_images_data_dict = {}
            for image_true_segment in total_image_segment_list:
                if image_true_segment.get() is None:
                    continue
                if image_true_segment.get()[0] not in total_images_data_dict:
                    total_images_data_dict[image_true_segment.get(
                    )[0]] = total_annotations_dict[image_true_segment.get()[0]]
                    total_images_data_dict[image_true_segment.get()[0]].true_segmentation_list.extend(
                        image_true_segment.get()[1])
                else:
                    total_images_data_dict[image_true_segment.get()[0]].true_segmentation_list.extend(
                        image_true_segment.get()[1])

            del total_annotations_dict, total_image_segment_list

            # 输出读取的source annotation至temp annotation
            process_temp_file_name_list = multiprocessing.Manager().list()
            process_output = multiprocessing.Manager().dict({'success_count': 0,
                                                            'fail_count': 0,
                                                             'no_segmentation': 0,
                                                             'temp_file_name_list': process_temp_file_name_list
                                                             })
            pool = multiprocessing.Pool(self.workers)
            for _, image in tqdm(total_images_data_dict.items()):
                pool.apply_async(func=self.output_temp_annotation(), args=(
                    self, image, process_output,),
                    error_callback=err_call_back)
            pool.close()
            pool.join()

            # 更新输出统计
            success_count += process_output['success_count']
            fail_count += process_output['fail_count']
            no_segmentation += process_output['no_segmentation']
            temp_file_name_list += process_output['temp_file_name_list']

        # 输出读取统计结果
        print('\nSource dataset convert to temp dataset file count: ')
        print('Total annotations:         \t {} '.format(
            len(os.listdir(self.source_dataset_annotations_folder))))
        print('Convert fail:              \t {} '.format(fail_count))
        print('No segmentation delete images: \t {} '.format(no_segmentation))
        print('Convert success:           \t {} '.format(success_count))
        self.temp_annotation_name_list = temp_file_name_list
        with open(os.path.join(self.temp_informations_folder, 'classes.names'), 'w') as f:
            if len(dataset['class_list_new']):
                f.write('\n'.join(str(n)
                                  for n in dataset['class_list_new']))
            f.close()

        return

    def transform_to_target_dataset():
        print('\nStart transform to target dataset:')

    def build_target_dataset_folder():
        print('\nStart build target dataset folder:')

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

    def load_image_annotation(self, one_annotation: dict, class_dict: dict, each_annotation_images_data_dict: dict) -> list:
        """[读取单个标签详细信息，并添加至each_annotation_images_data_dict]

        Args:
            dataset (dict): [数据集信息字典]
            one_annotation (dict): [单个数据字典信息]
            class_dict (dict): [description]
            process_output (dict): [each_annotation_images_data_dict进程间通信字典]

        Returns:
            list: [ann_image_id, true_box_list, true_segmentation_list]
        """

        ann_image_id = one_annotation['image_id']   # 获取此bbox图片id
        cls = class_dict[str(one_annotation['category_id'])]     # 获取bbox类别
        cls = cls.replace(' ', '').lower()
        if cls not in (dataset['source_segment_class_list']):
            return
        image = each_annotation_images_data_dict[ann_image_id]

        # 获取真实框信息
        true_box_list = []
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
            true_box_list.append(
                TRUE_BOX(cls, xmin, ymin, xmax, ymax))

        # 获取真实语义分割信息
        true_segmentation_list = []
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
                if '0' == one_annotation['iscrowd']:
                    true_segmentation_list.append(TRUE_SEGMENTATION(
                        cls, segment, area=one_annotation['area']))
                else:
                    true_segmentation_list.append(TRUE_SEGMENTATION(
                        cls, segment, one_annotation['area'], 1))

        return ann_image_id, true_box_list, true_segmentation_list

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
            dataset['temp_annotations_folder'],
            image.file_name_new + '.' + dataset['temp_annotation_form'])
        modify_true_box_list(image, dataset['detect_modify_class_dict'])
        modify_true_segmentation_list(
            image, dataset['segment_modify_class_dict'])
        if dataset['class_pixel_distance_dict'] is not None:
            class_box_pixel_limit(dataset, image.true_box_list)
            class_segmentation_pixel_limit(
                dataset, image.true_segmentation_list)
        if 0 == len(image.true_segmentation_list) and 0 == len(image.true_box_list):
            print('{} no true box and segmentation, has been delete.'.format(
                image.image_name_new))
            os.remove(image.image_path)
            process_output['no_detect_segmentation'] += 1
            process_output['fail_count'] += 1
            return
        if TEMP_OUTPUT(temp_annotation_output_path, image):
            process_output['temp_file_name_list'].append(image.file_name_new)
            process_output['success_count'] += 1
        else:
            process_output['fail_count'] += 1

        return
