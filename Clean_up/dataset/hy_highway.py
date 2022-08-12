'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-08-12 16:37:18
'''
import xml.etree.ElementTree as ET

from utils.utils import *
from base.image_base import *
from base.dataset_base import Dataset_Base


class HY_HIGHWAY(Dataset_Base):
    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_form = 'xml'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count(
        )

    def load_image_annotation(self, source_annotation_name: str,
                              process_output: dict) -> None:
        """将源标注转换为暂存标注

        Args:
            source_annotation_name (str): 源标注文件名称
            process_output (dict): 进程间通信字典
        """

        source_annotation_path = os.path.join(
            self.source_dataset_annotations_folder, source_annotation_name)
        tree = ET.parse(source_annotation_path)
        root = tree.getroot()
        image_name = os.path.splitext(str(
            root.find('filename').text))[0] + '.' + self.temp_image_form
        image_name_new = self.file_prefix + image_name
        image_path = os.path.join(self.temp_images_folder, image_name_new)
        img = cv2.imread(image_path)
        if img is None:
            print('Can not load: {}'.format(image_name_new))
            return
        height, width, channels = img.shape  # 读取每张图片的shape

        object_list = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            clss = str(obj.find('name').text)
            clss = clss.replace(' ', '').lower()
            if int(difficult) == 1:
                continue
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            xmin = max(min(float(b[0]), float(b[1]), float(width)), 0.)
            ymin = max(min(float(b[2]), float(b[3]), float(height)), 0.)
            xmax = min(max(float(b[1]), float(b[0]), 0.), float(width))
            ymax = min(max(float(b[3]), float(b[2]), 0.), float(height))
            box_xywh = [
                int(xmin),
                int(ymin),
                int(xmax - xmin),
                int(ymax - ymin)
            ]
            object_list.append(
                OBJECT(object_id=0,
                       object_clss=clss,
                       box_clss=clss,
                       box_xywh=box_xywh,
                       need_convert=self.need_convert))

        # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
        # 并将初始化后的对象存入total_images_data_list
        image = IMAGE(image_name, image_name_new, image_path, height, width,
                      channels, object_list)
        # 读取目标标注信息，输出读取的source annotation至temp annotation
        if image == None:
            return
        temp_annotation_output_path = os.path.join(
            self.temp_annotations_folder,
            image.file_name_new + '.' + self.temp_annotation_form)
        image.object_class_modify_and_pixel_limit(self)
        if 0 == len(image.object_list) and not self.keep_no_object:
            print('{} no object, has been delete.'.format(
                image.image_name_new))
            os.remove(image.image_path)
            process_output['no_object'] += 1
            return
        if image.output_temp_annotation(temp_annotation_output_path):
            process_output['temp_file_name_list'].append(image.file_name_new)
            process_output['success_count'] += 1
        else:
            print('errow output temp annotation: {}'.format(
                image.file_name_new))
            process_output['fail_count'] += 1

        return

    @staticmethod
    def target_dataset(dataset_instance: Dataset_Base) -> None:
        """[输出target annotation]

        Args:
            dataset (Dataset_Base): [数据集实例]
        """

        print('\nStart transform to target dataset:')

        return

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取HY_HIGHWAY数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成HY_HIGHWAY组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')

        return
