'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-10 18:00:02
'''
import time
import shutil
from PIL import Image
import multiprocessing

import dataset
from utils.utils import *
from base.image_base import *
from base.dataset_characteristic import *
from utils import image_form_transform
from base.dataset_base import Dataset_Base


class HY_VAL(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_form = 'json'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count()

    def source_dataset_copy_image_and_annotation(self):
        print('\nStart source dataset copy image and annotation:')
        pbar, update = multiprocessing_object_tqdm(
            self.source_dataset_image_count, 'Copy images')
        for root, _, files in os.walk(self.dataset_input_folder):
            pool = multiprocessing.Pool(self.workers)
            for n in files:
                if os.path.splitext(n)[-1].replace('.', '') in \
                        self.source_dataset_image_form_list:
                    pool.apply_async(self.source_dataset_copy_image,
                                     args=(root, n,),
                                     callback=update,
                                     error_callback=err_call_back)
            pool.close()
            pool.join()
        pbar.close()

        pbar, update = multiprocessing_object_tqdm(
            self.source_dataset_annotation_count, 'Copy annotations')
        for root, _, files in os.walk(self.dataset_input_folder):
            pool = multiprocessing.Pool(self.workers)
            for n in files:
                if os.path.splitext(n)[-1].replace('.', '') in \
                        self.source_dataset_image_form_list:
                    pool.apply_async(self.source_dataset_copy_annotation,
                                     args=(root, n,),
                                     callback=update,
                                     error_callback=err_call_back)
            pool.close()
            pool.join()
        pbar.close()

        print('Copy images and annotations end.')

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

        fake_js = {}
        if os.path.splitext(n)[-1].replace('.', '') in \
                self.source_dataset_image_form_list:
            json_name = os.path.splitext(
                n)[0] + '.' + self.source_dataset_annotation_form
            json_output_path = os.path.join(
                self.source_dataset_annotations_folder, json_name)
            json.dump(fake_js, open(json_output_path, 'w'))

        return

    def transform_to_temp_dataset(self) -> None:
        """[转换标注文件为暂存标注]
        """

        print('\nStart transform to temp dataset:')
        success_count = 0
        fail_count = 0
        no_object = 0
        temp_file_name_list = []

        for n in os.listdir(self.source_dataset_annotations_folder):
            source_dataset_annotation = os.path.join(
                self.source_dataset_annotations_folder, n)
            temp_annotation = os.path.join(
                self.temp_annotations_folder, self.file_prefix + n)
            shutil.copy(source_dataset_annotation, temp_annotation)
        return

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取HY_VAL数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        return []

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成HY_VAL组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')

        return
