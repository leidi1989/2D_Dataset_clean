'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-01-07 18:06:56
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
        for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
            pool = multiprocessing.Pool(dataset['workers'])
            for n in tqdm(files):
                if n.endswith(dataset['source_image_form']):
                    pool.apply_async(copy_image,
                                     args=(dataset, root, n,))
            pool.close()
            pool.join()
            print('Move images count: {}\n'.format(
                len(os.listdir(dataset['source_images_folder']))))

        print('Copy annotations: ')
        for root, dirs, files in tqdm(os.walk(dataset['source_path'])):
            pool = multiprocessing.Pool(dataset['workers'])
            for n in tqdm(files):
                if n.endswith(dataset['source_annotation_form']):
                    pool.apply_async(copy_annotation,
                                     args=(dataset, root, n,))
            pool.close()
            pool.join()
        print('Move annotations count: {}\n'.format(
            len(os.listdir(dataset['source_annotations_folder']))))

        return

    def transform_to_temp_dataset(self):
        print('\nStart transform to temp dataset:')

    def transform_to_target_dataset():
        print('\nStart transform to target dataset:')

    def build_target_dataset_folder():
        print('\nStart build target dataset folder:')

    def copy_image(self, dataset: dict, root: str, n: str) -> None:
        """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀]

        Args:
            dataset (dict): [数据集信息字典]
            root (str): [文件所在目录]
            n (str): [文件名]

        """

        image = os.path.join(root, n)
        temp_image = os.path.join(
            dataset['source_images_folder'], dataset['file_prefix'] + n)
        image_suffix = os.path.splitext(n)[-1].replace('.', '')
        if image_suffix != dataset['target_image_form']:
            dataset['transform_type'] = image_suffix + \
                '_' + dataset['target_image_form']
            image_form_transform.__dict__[
                dataset['transform_type']](image, temp_image)
            return
        else:
            shutil.copy(image, temp_image)
            return

    def copy_annotation(self, dataset: dict, root: str, n: str) -> None:
        """[复制源数据集标签文件至目标数据集中的source_annotations中]

        Args:
            dataset (dict): [数据集信息字典]
            root (str): [文件所在目录]
            n (str): [文件名]
        """

        annotation = os.path.join(root, n)
        temp_annotation = os.path.join(
            dataset['source_annotations_folder'], n)
        shutil.copy(annotation, temp_annotation)

        return
