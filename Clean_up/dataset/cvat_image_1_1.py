'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-14 10:37:00
'''
import time
import shutil
from PIL import Image
import multiprocessing
import xml.etree.ElementTree as ET

import dataset
from utils.utils import *
from base.image_base import *
from base.dataset_characteristic import *
from utils import image_form_transform
from base.dataset_base import Dataset_Base


class CVAT_IMAGE_1_1(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg']
        self.source_dataset_annotation_form = 'xml'
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
        total_source_dataset_annotations_list = os.listdir(
            self.source_dataset_annotations_folder)

        pbar, update = multiprocessing_list_tqdm(total_source_dataset_annotations_list,
                                                 desc='Total annotations')
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({'success_count': 0,
                                                         'fail_count': 0,
                                                         'no_object': 0,
                                                         'temp_file_name_list': process_temp_file_name_list
                                                         })
        pool = multiprocessing.Pool(self.workers)
        for source_annotation_name in total_source_dataset_annotations_list:
            pool.apply_async(func=self.load_annotation,
                             args=(source_annotation_name,
                                   process_output,),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        success_count = process_output['success_count']
        fail_count = process_output['fail_count']
        no_object = process_output['no_object']
        temp_file_name_list += process_output['temp_file_name_list']

        # 输出读取统计结果
        print('\nSource dataset convert to temp dataset file count: ')
        print('Total annotations:         \t {} '.format(
            len(total_source_dataset_annotations_list)))
        print('Convert fail:              \t {} '.format(
            fail_count))
        print('No object delete images: \t {} '.format(
            no_object))
        print('Convert success:           \t {} '.format(
            success_count))
        self.temp_annotation_name_list = temp_file_name_list
        print('Source dataset annotation transform to temp dataset end.')

        return

    def load_annotation(self, source_annotation_name: str, process_output: dict) -> None:
        """将源标注转换为暂存标注

        Args:
            source_annotation_name (str): 源标注文件名称
            process_output (dict): 进程间通信字典
        """

        source_annotations_path = os.path.join(
            self.source_dataset_annotations_folder, source_annotation_name)
        tree = ET.parse(source_annotations_path)
        root = tree.getroot()
        for annotation in root:
            if annotation.tag != 'image':
                continue
            image_name = str(annotation.attrib['name']).replace(
                '.' + self.source_dataset_image_form, '.' + self.target_dataset_image_form)
            image_name_new = self.file_prefix + image_name
            image_path = os.path.join(
                self.temp_images_folder, image_name_new)
            img = cv2.imread(image_path)
            if img is None:
                print('Can not load: {}'.format(image_name_new))
                return
            width = int(annotation.attrib['width'])
            height = int(annotation.attrib['height'])
            channels = img.shape[-1]
            object_list = []
            for n, obj in enumerate(annotation):
                cls = str(obj.attrib['label'])
                cls = cls.replace(' ', '').lower()
                segment = []
                for seg in obj.attrib['points'].split(';'):
                    x, y = seg.split(',')
                    x = float(x)
                    y = float(y)
                    segment.append(list(map(int, [x, y])))
                object_list.append(OBJECT(n,
                                          cls,
                                          segmentation_clss=cls,
                                          segmentation=segment))
            image = IMAGE(image_name, image_name_new, image_path,
                          height, width, channels, object_list)
            # 读取目标标注信息，输出读取的source annotation至temp annotation
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
                process_output['temp_file_name_list'].append(
                    image.file_name_new)
                process_output['success_count'] += 1
            else:
                print('errow output temp annotation: {}'.format(
                    image.file_name_new))
                process_output['fail_count'] += 1

            return

    @staticmethod
    def target_dataset(dataset_instance: object):
        """[输出temp dataset annotation]

        Args:
            dataset (Dataset): [dataset]
        """

        print('\nStart transform to target dataset:')
        encode = []
        for n in range(len(dataset['class_list_new'])):
            encode.append(random.randint(0, 255))
        # 转换不重复随机颜色编码为16进制颜色
        color_list = []
        for n in encode:
            color_list.append(RGB_to_Hex(str(n)+','+str(n)+','+str(n)))

        # 生成空基本信息xml文件
        annotations = dataset.__dict__[
            dataset_instance.target_dataset_style].annotation_creat_root(dataset, color_list)
        # 获取全部图片标签信息列表
        total_image_xml = []
        pbar, update = multiprocessing_list_tqdm(dataset_instance.temp_annotation_path_list,
                                                 desc='transform to target dataset')
        pool = multiprocessing.Pool(dataset_instance.workers)
        for temp_annotation_path in dataset_instance.temp_annotation_path_list:
            total_image_xml.append(pool.apply_async(func=dataset.__dict__[dataset_instance.target_dataset_style].annotation_get_temp,
                                                    args=(dataset_instance,
                                                          temp_annotation_path,),
                                                    callback=update,
                                                    error_callback=err_call_back))
        pool.close()
        pool.join()
        pbar.close()

        # 将image标签信息添加至annotations中
        for n, image in enumerate(total_image_xml):
            annotation = image.get()
            annotation.attrib['id'] = str(n)
            annotations.append(annotation)

        tree = ET.ElementTree(annotations)

        annotation_output_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder,
            'annotatons.' + dataset_instance.target_dataset_annotation_form)
        tree.write(annotation_output_path,
                   encoding='utf-8', xml_declaration=True)

        return

    @staticmethod
    def annotation_creat_root(dataset_instance: dict, color_list: list) -> None:
        """[创建xml根节点]

        Args:
            dataset (dict): [数据集信息字典]
            color_list (list): [色彩列表]

        Returns:
            [type]: [xml根节点]
        """

        class_id = 0
        annotations = ET.Element('annotations')
        version = ET.SubElement(annotations, 'version')
        version.text = '1.1'
        meta = ET.SubElement(annotations, 'meta')
        task = ET.SubElement(meta, 'task')
        # ET.SubElement(task, 'id')
        # ET.SubElement(task, 'name')
        # ET.SubElement(task, 'size')
        # mode = ET.SubElement(task, 'mode')
        # mode.text = 'annotation'
        # overlap = ET.SubElement(task, 'overlap')
        # ET.SubElement(task, 'bugtracker')
        # ET.SubElement(task, 'created')
        # ET.SubElement(task, 'updated')
        # subset = ET.SubElement(task, 'subset')
        # subset.text = 'default'
        # start_frame = ET.SubElement(task, 'start_frame')
        # start_frame.text='0'
        # ET.SubElement(task, 'stop_frame')
        # ET.SubElement(task, 'frame_filter')
        # segments = ET.SubElement(task, 'segments')
        # segment = ET.SubElement(segments, 'segment')
        # ET.SubElement(segments, 'id')
        # start = ET.SubElement(segments, 'start')
        # start.text='0'
        # ET.SubElement(segments, 'stop')
        # ET.SubElement(segments, 'url')
        # owner = ET.SubElement(task, 'owner')
        # ET.SubElement(owner, 'username')
        # ET.SubElement(owner, 'email')
        # ET.SubElement(task, 'assignee')
        labels = ET.SubElement(task, 'labels')

        class_dict_list_output_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder, 'class_dict_list.txt')
        with open(class_dict_list_output_path, 'w') as f:
            for n, c in zip(dataset['class_list_new'], color_list):
                label = ET.SubElement(labels, 'label')
                name = ET.SubElement(label, 'name')
                name.text = n
                color = ET.SubElement(label, 'color')
                color.text = c
                attributes = ET.SubElement(label, 'attributes')
                attribute = ET.SubElement(attributes, 'attribute')
                name = ET.SubElement(attribute, 'name')
                name.text = '1'
                mutable = ET.SubElement(attribute, 'mutable')
                mutable.text = 'False'
                input_type = ET.SubElement(attribute, 'input_type')
                input_type.text = 'text'
                default_value = ET.SubElement(attribute, 'default_value')
                default_value.text = None
                values = ET.SubElement(attribute, 'values')
                values.text = None
                # 输出标签色彩txt
                s = '  {\n    "name": "'+n+'",\n    "color": "' + \
                    str(c)+'",\n    "attributes": []\n  },\n'
                f.write(s)
                class_id += 1

            # ET.SubElement(task, 'dumped')
        return annotations

    @staticmethod
    def annotation_get_temp(dataset_instance: dict, temp_annotation_path: str) -> None:
        """[获取temp标签信息]

        Args:
            dataset (dict): [数据集信息字典]
            temp_annotation_path (str): [暂存标签路径]
        """

        image = dataset_instance.TEMP_LOAD(
            dataset_instance, temp_annotation_path)
        if image == None:
            return
        image_xml = ET.Element('image', {
            'id': '', 'name': image.image_name_new, 'width': str(image.width), 'height': str(image.height)})
        for n in image.true_segmentation_list:
            point_list = []
            for x in n.segmentation:
                point_list.append(str(x[0])+','+str(x[1]))
            if 2 == len(point_list):
                continue
            polygon = ET.SubElement(image_xml, 'polygon', {
                                    'label': n.clss, 'occluded': '0', 'source': 'manual', 'points': ';'.join(point_list)})
            attribute = ET.SubElement(polygon, 'attribute', {'name': '1'})

        return image_xml

    @staticmethod
    def annotation_check(dataset_instance: object) -> list:
        """[读取COCO2017数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []
        dataset_instance.target_check_file_name_list = os.listdir(
            dataset_instance.target_dataset_annotations_folder)  # 读取target_annotations_folder文件夹下的全部文件名
        images_data_list = []
        images_data_dict = {}
        for target_annotation in dataset_instance.target_check_file_name_list:
            if target_annotation != 'instances_train2017.json':
                continue
            target_annotation_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder, target_annotation)
            print('Loading instances_train2017.json:')
            with open(target_annotation_path, 'r') as f:
                data = json.loads(f.read())
            name_dict = {}
            for one_name in data['categories']:
                name_dict['%s' % one_name['id']] = one_name['name']

            print('Start count images:')
            total_image_count = 0
            for d in tqdm(data['images']):
                total_image_count += 1
            check_images_count = min(
                dataset_instance.target_dataset_annotations_check_count, total_image_count)
            check_image_id_list = [random.randint(
                0, total_image_count)for i in range(check_images_count)]

            print('Start load each annotation data file:')
            for n in check_image_id_list:
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

            for one_annotation in tqdm(data['annotations']):
                if one_annotation['image_id'] in images_data_dict:
                    ann_image_id = one_annotation['image_id']   # 获取此bbox图片id
                    box_xywh = []
                    segmentation = []
                    segmentation_area = None
                    segmentation_iscrowd = 0
                    keypoints_num = 0
                    keypoints = []
                    cls = name_dict[str(one_annotation['category_id'])]
                    cls = cls.replace(' ', '').lower()
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

                    one_object = OBJECT(id, cls, cls, cls, cls,
                                        box_xywh, segmentation, keypoints_num, keypoints,
                                        dataset_instance.task_convert,
                                        segmentation_area=segmentation_area,
                                        segmentation_iscrowd=segmentation_iscrowd
                                        )
                    images_data_dict[ann_image_id].object_list.append(
                        one_object)

        for _, n in images_data_dict.items():
            images_data_list.append(n)
        random.shuffle(images_data_list)
        check_images_count = min(
            dataset_instance.target_dataset_annotations_check_count, len(images_data_list))
        check_images_list = images_data_list[0:check_images_count]

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: object) -> None:
        """[生成COCO 2017组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')
        # 调整image
        output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder, 'coco2017'))
        shutil.rmtree(output_root)
        output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder, 'coco2017'))
        annotations_output_folder = check_output_path(
            os.path.join(output_root, 'annotations'))
        # 调整ImageSets
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
