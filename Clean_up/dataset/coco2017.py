'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-01-07 19:13:27
'''
import shutil
import multiprocessing

from utils.utils import *
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

    def transform_to_temp_dataset(self):
        print('\nStart transform to temp dataset:')

    def transform_to_target_dataset():
        print('\nStart transform to target dataset:')

    def build_target_dataset_folder():
        print('\nStart build target dataset folder:')
