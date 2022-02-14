'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-14 18:27:21
'''
import shutil
from PIL import Image
import multiprocessing

import dataset
from utils.utils import *
from base.image_base import *
from base.dataset_characteristic import *
from utils import image_form_transform
from base.dataset_base import Dataset_Base
from utils.convertion_function import yolo, revers_yolo


class YOLO(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_form = 'txt'
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

        pbar, update = multiprocessing_object_tqdm(
            self.source_dataset_annotation_count, 'Total annotations')
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({'success_count': 0,
                                                         'fail_count': 0,
                                                         'no_object': 0,
                                                         'temp_file_name_list': process_temp_file_name_list
                                                         })
        pool = multiprocessing.Pool(self.workers)
        for source_annotation_name in os.listdir(self.source_dataset_annotations_folder):
            pool.apply_async(func=self.load_image_annotation,
                             args=(source_annotation_name,
                                   process_output,),
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

    def load_image_annotation(self, source_annotation_name: str, process_output: dict) -> list:
        """[读取单个标签详细信息, 并添加至each_annotation_images_data_dict]

        Args:
            id(int): [标注id]
            dataset (dict): [数据集信息字典]
            one_annotation (dict): [单个数据字典信息]
            class_dict (dict): [类别字典]
            process_output (dict): [each_annotation_images_data_dict进程间通信字典]

        Returns:
            list: [ann_image_id, true_box_list, true_segmentation_list]
        """

        source_annotation_path = os.path.join(
            self.source_dataset_annotations_folder, source_annotation_name)
        with open(source_annotation_path, 'r') as f:
            data = json.loads(f.read())

        del f

        class_dict = {}
        for n in data['categories']:
            class_dict['%s' % n['id']] = n['name']

        image_name = os.path.splitext(data['images'][0]['file_name'])[
            0] + '.' + self.temp_image_form
        image_name_new = self.file_prefix + image_name
        image_path = os.path.join(
            self.temp_images_folder, image_name_new)
        img = Image.open(image_path)
        height, width = img.height, img.width
        channels = 3
        # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
        # 并将初始化后的对象存入total_images_data_list
        object_list = []
        for one_annotation in data['annotations']:
            id = one_annotation['id']
            box_xywh = []
            segmentation = []
            segmentation_area = None
            segmentation_iscrowd = 0
            keypoints_num = 0
            keypoints = []
            cls = class_dict[str(one_annotation['category_id'])]
            cls = cls.replace(' ', '').lower()
            total_class = []
            for _, task_class_dict in self.task_dict.items():
                if task_class_dict is None:
                    continue
                total_class.extend(task_class_dict['Source_dataset_class'])
            if cls not in total_class:
                continue
            # 获取真实框信息
            if 'bbox' in one_annotation and len(one_annotation['bbox']):
                box = [one_annotation['bbox'][0],
                       one_annotation['bbox'][1],
                       one_annotation['bbox'][0] + one_annotation['bbox'][2],
                       one_annotation['bbox'][1] + one_annotation['bbox'][3]]
                xmin = max(min(int(box[0]), int(box[2]),
                               int(width)), 0.)
                ymin = max(min(int(box[1]), int(box[3]),
                               int(height)), 0.)
                xmax = min(max(int(box[2]), int(box[0]), 0.),
                           int(width))
                ymax = min(max(int(box[3]), int(box[1]), 0.),
                           int(height))
                box_xywh = [xmin, ymin, xmax-xmin, ymax-ymin]

            # 获取真实语义分割信息
            if 'segmentation' in one_annotation and len(one_annotation['segmentation']):
                segment = []
                point = []
                for i, x in enumerate(one_annotation['segmentation']):
                    if 0 == i % 2:
                        point.append(x)
                    else:
                        point.append(x)
                        point = list(map(int, point))
                        segment.append(point)
                        if 2 != len(point):
                            print('Segmentation label wrong: ', image_name_new)
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

            object_list.append(OBJECT(id, cls, cls, cls, cls,
                                      box_xywh, segmentation, keypoints_num, keypoints,
                                      self.task_convert,
                                      segmentation_area=segmentation_area,
                                      segmentation_iscrowd=segmentation_iscrowd,
                                      ))
        image = IMAGE(image_name, image_name_new,
                      image_path, height, width, channels, object_list)

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
        """[输出target annotation]

        Args:
            dataset (object): [数据集类]
        """

        print('\nStart transform to target dataset:')
        total_annotation_path_list = []
        for dataset_temp_annotation_path_list in tqdm(dataset_instance.temp_divide_file_list[1:-1],
                                                      desc='Get total annotation path list'):
            with open(dataset_temp_annotation_path_list, 'r') as f:
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
        """读取暂存annotation

        Args:
            dataset_instance (object): 数据集实例
            temp_annotation_path (str): 暂存annotation路径
        """

        image = dataset_instance.TEMP_LOAD(
            dataset_instance, temp_annotation_path)
        if image == None:
            return
        annotation_output_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder,
            image.file_name + '.' + dataset_instance.target_dataset_annotation_form)
        one_image_bbox = []                                     # 声明每张图片bbox列表
        for true_box in image.true_box_list:                        # 遍历单张图片全部bbox
            true_box_class = str(true_box.clss).replace(
                ' ', '').lower()    # 获取bbox类别
            if true_box_class in set(dataset['class_list_new']):
                cls_id = dataset['class_list_new'].index(true_box_class)
                b = (true_box.xmin, true_box.xmax, true_box.ymin,
                     true_box.ymax,)                                # 获取源标签bbox的xxyy
                bb = yolo((image.width, image.height),
                          b)       # 转换bbox至yolo类型
                one_image_bbox.append([cls_id, bb])
            else:
                print('\nErro! Class not in classes.names image: %s!' %
                      image.image_name)

        with open(annotation_output_path, 'w') as f:   # 创建图片对应txt格式的label文件
            for one_bbox in one_image_bbox:
                f.write(str(one_bbox[0]) + " " +
                        " ".join([str(a) for a in one_bbox[1]]) + '\n')
            f.close()

        return

    @staticmethod
    def annotation_check(dataset_instance: object) -> list:
        """[读取YOLO数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []
        dataset_instance.target_check_file_name_list = annotations_path_list(
            dataset_instance.total_file_name_path, dataset_instance.target_dataset_annotation_check_count)
        for n in dataset_instance.target_check_file_name_list:
            target_annotation_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder,
                n + '.' + dataset_instance.target_dataset_annotation_form)
            with open(target_annotation_path, 'r') as f:
                image_name = n + '.' + dataset['target_image_form']
                image_path = os.path.join(
                    dataset_instance.temp_images_folder, image_name)
                img = cv2.imread(image_path)
                size = img.shape
                width = int(size[1])
                height = int(size[0])
                channels = int(size[2])
                object_list = []
                for one_bbox in f.read().splitlines():
                    true_box = one_bbox.split(' ')[1:]
                    cls = dataset['class_list_new'][int(
                        one_bbox.split(' ')[0])]
                    cls = cls.strip(' ').lower()
                    if cls not in dataset['class_list_new']:
                        continue
                    true_box = revers_yolo(size, true_box)
                    xmin = min(
                        max(min(float(true_box[0]), float(true_box[1])), 0.), float(width))
                    ymin = min(
                        max(min(float(true_box[2]), float(true_box[3])), 0.), float(height))
                    xmax = max(
                        min(max(float(true_box[1]), float(true_box[0])), float(width)), 0.)
                    ymax = max(
                        min(max(float(true_box[3]), float(true_box[2])), float(height)), 0.)
                    object_list.append(OBJECT(
                        cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
                image = IMAGE(image_name, image_name, image_path, int(
                    height), int(width), int(channels), object_list)
                check_images_list.append(image)

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: object) -> None:
        """[生成YOLO组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
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
