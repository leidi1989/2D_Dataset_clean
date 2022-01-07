'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-01-07 18:02:34
'''
from utils.utils import *
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
                    pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_image,
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
                    pool.apply_async(F.__dict__[dataset['source_dataset_stype']].copy_annotation,
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
        