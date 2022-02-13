'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-13 14:04:52
'''
import shutil
import multiprocessing

import dataset
from utils.utils import *
from base.image_base import *
from base.dataset_characteristic import *
from utils import image_form_transform
from base.dataset_base import Dataset_Base


class CITYSCAPES_VAL(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_form = 'json'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count()

    def source_dataset_copy_image_and_annotation(self) -> None:
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
                if n.endswith(self.source_dataset_annotation_form):
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

        annotation = os.path.join(root, n)
        temp_annotation = os.path.join(
            self.source_dataset_annotations_folder, n)
        shutil.copy(annotation, temp_annotation)

        return

    def transform_to_temp_dataset(self) -> None:
        """[转换标注文件为暂存标注]
        """

        print('\nStart transform to temp dataset:')
        success_count = 0
        fail_count = 0
        no_object = 0
        temp_file_name_list = []

        return

    @staticmethod
    def target_dataset(dataset_instance: object) -> None:
        """[输出target annotation]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart transform to target dataset:')
        total_annotation_path_list = []

        with open(dataset_instance.temp_divide_file_list[0], 'r') as f:
            for n in f.readlines():
                total_annotation_path_list.append(n.replace('\n', '')
                                                  .replace(dataset_instance.source_dataset_images_folder,
                                                           dataset_instance.temp_annotations_folder)
                                                  .replace(dataset_instance.target_dataset_image_form,
                                                           dataset_instance.temp_annotation_form))

        pbar, update = multiprocessing_list_tqdm(
            total_annotation_path_list, desc='Output target dataset annotation')
        pool = multiprocessing.Pool(dataset_instance.workers)
        for temp_annotation_path in total_annotation_path_list:
            pool.apply_async(func=dataset.__dict__[dataset_instance.target_dataset_style].annotation_output,
                             args=(dataset_instance,
                                   temp_annotation_path,),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        return

    @staticmethod
    def annotation_output(dataset_instance: object, temp_annotation_path: str) -> None:
        """[读取暂存annotation]

        Args:
            dataset_instance (object): [数据集信息字典]
            temp_annotation_path (str): [annotation路径]
        """

        file = os.path.splitext(temp_annotation_path.split(os.sep)[-1])[0]
        annotation_output_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder,
            file + '.' + dataset_instance.target_dataset_annotation_form)
        shutil.copy(temp_annotation_path, annotation_output_path)

        return

    @staticmethod
    def annotation_check(dataset_instance: object) -> list:
        """[读取CITYSCAPES数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        return []

    @staticmethod
    def target_dataset_folder(dataset_instance: object) -> None:
        """[生成CITYSCAPES_VAL组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')
        # 获取全量数据编号字典
        file_name_dict = {}
        print('Collect file name dict.')
        for x, n in enumerate(sorted(os.listdir(dataset_instance.temp_images_folder))):
            file_name = os.path.splitext(n.split(os.sep)[-1])[0]
            file_name_dict[file_name] = x

        output_root = check_output_path(os.path.join(
            dataset_instance.dataset_output_folder, 'cityscapes', 'data'))   # 输出数据集文件夹
        cityscapes_folder_list = ['gtFine', 'leftImg8bit']
        data_divion_name = ['train', 'test', 'val']
        output_folder_path_list = []
        # 创建cityscapes组织结构
        print('Clean dataset folder!')
        shutil.rmtree(output_root)
        print('Create new folder:')
        for n in cityscapes_folder_list:
            output_folder_path = check_output_path(
                os.path.join(output_root, n))
            output_folder_path_list.append(output_folder_path)
            for m in data_divion_name:
                dataset_division_folder_path = os.path.join(
                    output_folder_path, m)
                check_output_path(dataset_division_folder_path)
                check_output_path(os.path.join(
                    dataset_division_folder_path,
                    dataset_instance.file_prefix.replace(
                        dataset_instance.file_prefix_delimiter, '')))

        print('Create annotation file to output folder:')
        for x in tqdm(os.listdir(dataset_instance.temp_images_folder),
                      desc='Create annotation file to folder:'):
            file = os.path.splitext(x.split(os.sep)[-1])[0]
            file_out = dataset_instance.file_prefix.replace(
                dataset_instance.file_prefix_delimiter, '') + '_000000_' + \
                str(format(file_name_dict[file], '06d'))
            # 调整image
            image_out = file_out + '_leftImg8bit' + \
                '.' + dataset_instance.target_dataset_image_form
            image_path = os.path.join(
                dataset_instance.temp_images_folder,
                file + '.' + dataset_instance.target_dataset_image_form)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue

            image_output_path = os.path.join(
                output_folder_path_list[1], 'val', dataset_instance.file_prefix.replace(
                    dataset_instance.file_prefix_delimiter, ''), image_out)
            # 调整annotation
            annotation_out = file_out + '_gtFine_polygons' + \
                '.' + 'json'
            annotation_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder,
                file + '.' + dataset_instance.target_dataset_annotation_form)
            annotation_output_path = os.path.join(
                output_folder_path_list[0], 'val', dataset_instance.file_prefix.replace(
                    dataset_instance.file_prefix_delimiter, ''), annotation_out)
            # 调整annotation为_gtFine_labelIds.png
            labelIds_out = file_out + '_gtFine_labelIds.png'
            labelIds_output_path = os.path.join(
                output_folder_path_list[0], 'val', dataset_instance.file_prefix.replace(
                    dataset_instance.file_prefix_delimiter, ''), labelIds_out)
            # 输出
            shutil.copy(image_path, image_output_path)
            shutil.copy(annotation_path, annotation_output_path)

            img = image.shape
            zeros = np.zeros((img[0], img[1]), dtype=np.uint8)
            cv2.imwrite(labelIds_output_path, zeros)

        return
