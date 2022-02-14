'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-14 17:52:55
'''
import shutil
import multiprocessing

import dataset
from utils.utils import *
from base.image_base import *
from base.dataset_characteristic import *
from utils import image_form_transform
from base.dataset_base import Dataset_Base
from utils.convertion_function import temp_box_to_coco_box


class BDD100K(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg']
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
            pool.apply_async(func=self.load_image_annotation,
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

    def load_image_annotation(self, source_annotation_name: str, process_output: dict) -> None:
        """将源标注转换为暂存标注

        Args:
            source_annotation_name (str): 源标注文件名称
            process_output (dict): 进程间通信字典
        """

        area_list = ['area/drivable',
                     'area/alternative',
                     'area/unknown'
                     ]
        start_point_dist_threshhold = {'lane/crosswalk': 800,
                                       'lane/doubleother': 300,
                                       'lane/doublewhite': 300,
                                       'lane/doubleyellow': 300,
                                       'lane/roadcurb': 50,
                                       'lane/singleother': 70,
                                       'lane/singlewhite': 45,
                                       'lane/singleyellow': 70}
        dist_var_threshhold = {'lane/crosswalk': 5000,
                               'lane/doubleother': 5000,
                               'lane/doublewhite': 5000,
                               'lane/doubleyellow': 5000,
                               'lane/roadcurb': 5000,
                               'lane/singleother': 5000,
                               'lane/singlewhite': 5000,
                               'lane/singleyellow': 5000}
        one_line_expand_offset = 5

        source_annotation_path = os.path.join(
            self.source_dataset_annotations_folder, source_annotation_name)
        with open(source_annotation_path, 'r') as f:
            data = json.loads(f.read())
        object_list = []
        # 获取data字典中images内的图片信息，file_name、height、width
        image_name = os.path.splitext(source_annotation_name)[
            0] + '.' + self.temp_image_form
        image_name_new = self.file_prefix + image_name

        # TODO debug
        # if image_name_new != 'bdd100k@00db9030-5102ed41.png':
        #     return

        image_path = os.path.join(
            self.temp_images_folder, image_name_new)
        img = cv2.imread(image_path)
        height, width, channels = img.shape

        # 标注按定义类别分类
        object_box_list = []
        object_segment_area_list = []
        object_segment_lane_list = []
        object_count = 0

        for object in data['frames'][0]['objects']:
            if 'box2d' in object:
                object_box_list.append(object)
            if 'poly2d' in object:
                clss = object['category']
                clss = clss.replace(' ', '').lower()
                if clss in area_list:
                    object_segment_area_list.append(object)
                else:
                    if object['attributes']['direction'] == 'vertical':
                        continue
                    object_segment_lane_list.append(object)

        # true box
        for n, object in enumerate(object_box_list):
            clss = object['category']
            clss = clss.replace(' ', '').lower()
            box_xywh = temp_box_to_coco_box([object['box2d']['x1'],
                                             object['box2d']['y1'],
                                             object['box2d']['x2'],
                                             object['box2d']['y2']])
            object_list.append(OBJECT(n,
                                      clss,
                                      box_clss=clss,
                                      box_xywh=box_xywh,
                                      box_color=object['attributes']['trafficLightColor']))
        object_count += len(object_box_list)

        # object segment area
        for n, object in enumerate(object_segment_area_list):
            clss = object['category']
            clss = clss.replace(' ', '').lower()
            segmentation_point_list = []
            last_point = ''
            temp_point = []
            c_count = 0
            # 三阶贝塞尔曲线解算
            for point in object['poly2d']:
                if point[2] == 'L':
                    if '' == last_point:
                        segmentation_point_list.append(point[0:-1])
                        temp_point.append(point[0:-1])
                        last_point = 'L'
                    elif 'L' == last_point:
                        segmentation_point_list += temp_point
                        temp_point = []
                        temp_point.append(point[0:-1])
                        last_point = 'L'
                else:
                    temp_point.append(point[0:-1])
                    last_point = 'C'
                    c_count += 1
                    if 3 == c_count:
                        segmentation_point_list.append(temp_point[0])
                        bezier_line = []
                        for r in range(1, 21):
                            r = r / 20
                            bezier_line.append(calNextPoints(
                                temp_point, rate=r)[0])
                        segmentation_point_list += bezier_line
                        temp_point = [temp_point[-1]]
                        last_point = 'L'
                        c_count = 0
            segmentation_point_list = np.array(segmentation_point_list)
            segmentation_point_list = np.maximum(segmentation_point_list, 0)
            segmentation_point_list[:, 0] = np.minimum(
                segmentation_point_list[:, 0], 1280)
            segmentation_point_list[:, 1] = np.minimum(
                segmentation_point_list[:, 1], 720)
            segmentation_point_list_temp = []
            for n in segmentation_point_list:
                segmentation_point_list_temp.append(n.astype(np.int).tolist())
            segmentation_point_list = segmentation_point_list_temp
            object_list.append(OBJECT(n+object_count,
                                      clss,
                                      segmentation_clss=clss,
                                      segmentation=segmentation_point_list))
        object_count += len(object_segment_area_list)

        # 车道线提取，lane单双线标注分类
        object_segment_one_line_lane_list = []
        object_segment_double_line_lane_pair_list = []
        for lane in object_segment_lane_list:
            # 对贝塞尔标注线段进行解析并按min_y的x坐标进行排序
            if 2 == len(lane['poly2d']):
                segmentation_point_list = [x[0:-1] for x in lane['poly2d']]
                line_point_list = [lane['poly2d'][0][0:-1]]
                line_point_list.append(lane['poly2d'][1][0:-1])
                line_point_list.sort(key=lambda line_point: (
                    line_point[1], -line_point[0]), reverse=True)
                lane.update({'line_point_list': line_point_list})
                lane.update({'zero_to_start_point_dist': dist(
                    np.array(lane['line_point_list'][0]), np.array([0, 0]))})
            else:
                segmentation_point_list = [x[0:-1] for x in lane['poly2d']]
                line_point_list = [lane['poly2d'][0][0:-1]]
                for r in range(1, 21):
                    r = r / 20
                    line_point_list.append(calNextPoints(
                        segmentation_point_list, rate=r)[0])
                line_point_list.sort(key=lambda line_point: (
                    line_point[1], -line_point[0]))
                lane.update({'line_point_list': line_point_list})
                lane.update({'zero_to_start_point_dist': dist(
                    np.array(lane['line_point_list'][0]), np.array([0, 0]))})
        # 对标注线段按起始点到[0, 0]点距离进行排序
        object_segment_lane_list.sort(
            key=lambda x: x['zero_to_start_point_dist'])

        # 对线段进行类别划分
        lane_class_dict = {}
        for line in object_segment_lane_list:
            if line['category'] not in lane_class_dict:
                lane_class_dict.update({line['category']: [line]})
            else:
                lane_class_dict[line['category']].append(line)

        # 对进行类别划分后的车道线按单双线标注进行分类
        temp_line = {}
        for key, value in lane_class_dict.items():
            for line in value:
                if not temp_line:
                    temp_line = line
                else:
                    if temp_line['category'] != key:
                        object_segment_one_line_lane_list.append(temp_line)
                        temp_line = line
                    else:
                        line_points_dist = []
                        for m, n in zip(temp_line['line_point_list'], line['line_point_list']):
                            line_points_dist.append(
                                dist(np.array(m), np.array(n)))
                        line_points_dist = np.array(line_points_dist)
                        line_points_dist_var = np.var(line_points_dist)
                        line_points_dist_mean = np.mean(line_points_dist)
                        lines_start_point_dist = dist(np.array(
                            temp_line['line_point_list'][0]), np.array(line['line_point_list'][0]))
                        lines_end_point_dist = dist(np.array(
                            temp_line['line_point_list'][-1]), np.array(line['line_point_list'][-1]))
                        if (lines_start_point_dist <= start_point_dist_threshhold[key.replace(' ', '')] or
                            lines_end_point_dist <= start_point_dist_threshhold[key.replace(' ', '')]) \
                                and line_points_dist_mean <= start_point_dist_threshhold[key.replace(' ', '')]*2 \
                                and line_points_dist_var <= dist_var_threshhold[key.replace(' ', '')]:
                            object_segment_double_line_lane_pair_list.append(
                                [temp_line, line])
                            temp_line = {}
                        else:
                            object_segment_one_line_lane_list.append(temp_line)
                            temp_line = line

        # object segment double line lane
        for p, q in enumerate(object_segment_double_line_lane_pair_list):
            m = q[0]
            n = q[1]
            clss = m['category']
            clss = clss.replace(' ', '').lower()
            # line 1
            segmentation_point_list = [x[0:-1] for x in m['poly2d']]
            line_point_list_1 = [m['poly2d'][0][0:-1]]
            for r in range(1, 21):
                r = r / 20
                line_point_list_1.append(calNextPoints(
                    segmentation_point_list, rate=r)[0])
            # line 2
            segmentation_point_list = [x[0:-1] for x in n['poly2d']]
            line_point_list_2 = [n['poly2d'][0][0:-1]]
            for r in range(1, 21):
                r = r / 20
                line_point_list_2.append(calNextPoints(
                    segmentation_point_list, rate=r)[0])

            pair_line_dist_0_0 = dist(
                np.array(line_point_list_1[0]), np.array(line_point_list_2[0]))
            pair_line_dist_0_1 = dist(
                np.array(line_point_list_1[0]), np.array(line_point_list_2[-1]))
            if pair_line_dist_0_0 <= pair_line_dist_0_1:
                line_point_list_2.reverse()
            line_point_list_1 += line_point_list_2
            line_point_list_1 = np.array(line_point_list_1)
            line_point_list_1 = np.maximum(line_point_list_1, 0)
            line_point_list_1[:, 0] = np.minimum(line_point_list_1[:, 0], 1280)
            line_point_list_1[:, 1] = np.minimum(line_point_list_1[:, 1], 720)
            line_point_list_1_temp = []
            for n in line_point_list_1:
                line_point_list_1_temp.append(n.astype(np.int).tolist())
            line_point_list_1 = line_point_list_1_temp
            object_list.append(OBJECT(p+object_count,
                                      clss,
                                      segmentation_clss=clss,
                                      segmentation=line_point_list_1))
        object_count += len(object_segment_double_line_lane_pair_list)

        # object segment one line lane
        for n, object in enumerate(object_segment_one_line_lane_list):
            clss = object['category']
            clss = clss.replace(' ', '').lower()
            segmentation_point_list = [x[0:-1] for x in object['poly2d']]
            line_point_list = [object['poly2d'][0][0:-1]]
            # 直线
            if 2 == len(segmentation_point_list):
                line_point_list_1 = [[x - one_line_expand_offset for x in object['poly2d'][0][0:-1]],
                                     [x - one_line_expand_offset for x in object['poly2d'][1][0:-1]]]

                line_point_list_2 = [[x + one_line_expand_offset for x in object['poly2d'][0][0:-1]],
                                     [x + one_line_expand_offset for x in object['poly2d'][1][0:-1]]]

                line_point_list_l = np.array(line_point_list_1)
                line_point_list_r = np.flipud(np.array(line_point_list_2))
                line_point_list_loop = np.append(
                    line_point_list_l, line_point_list_r, axis=0)
                line_point_list_loop = np.maximum(line_point_list_loop, 0)
                line_point_list_loop[:, 0] = np.minimum(
                    line_point_list_loop[:, 0], 1280)
                line_point_list_loop[:, 1] = np.minimum(
                    line_point_list_loop[:, 1], 720)
                line_point_list_loop_temp = []
                for n in line_point_list_loop:
                    line_point_list_loop_temp.append(n.astype(np.int).tolist())
                line_point_list_loop = line_point_list_loop_temp
                object_list.append(OBJECT(n+object_count,
                                          clss,
                                          segmentation_clss=clss,
                                          segmentation=line_point_list_loop))
            # 贝塞尔曲线
            else:
                # 单线左侧边缘
                line_point_list_1 = [[
                    x - one_line_expand_offset for x in object['poly2d'][0][0:-1]]]
                line_point_list_1_c = []
                for points in object['poly2d']:
                    line_point_list_1_c.append(
                        [points[0:-1][0] - one_line_expand_offset, points[0:-1][1]])
                for r in range(1, 21):
                    r = r / 20
                    line_point_list_1.append(calNextPoints(
                        line_point_list_1_c, rate=r)[0])
                # 单线右侧边缘
                line_point_list_2 = [[
                    x - one_line_expand_offset for x in object['poly2d'][0][0:-1]]]
                line_point_list_2_c = []
                for points in object['poly2d']:
                    line_point_list_2_c.append(
                        [points[0:-1][0] + one_line_expand_offset, points[0:-1][1]])
                for r in range(1, 21):
                    r = r / 20
                    line_point_list_2.append(calNextPoints(
                        line_point_list_2_c, rate=r)[0])

                line_point_list_l = np.array(line_point_list_1)
                line_point_list_r = np.flipud(np.array(line_point_list_2))
                line_point_list_loop = np.append(
                    line_point_list_l, line_point_list_r, axis=0)
                line_point_list_loop = np.maximum(line_point_list_loop, 0)
                line_point_list_loop[:, 0] = np.minimum(
                    line_point_list_loop[:, 0], 1280)
                line_point_list_loop[:, 1] = np.minimum(
                    line_point_list_loop[:, 1], 720)
                line_point_list_loop_temp = []
                for n in line_point_list_loop:
                    line_point_list_loop_temp.append(n.astype(np.int).tolist())
                line_point_list_loop = line_point_list_loop_temp
                object_list.append(OBJECT(n+object_count,
                                          clss,
                                          segmentation_clss=clss,
                                          segmentation=line_point_list_loop))
        object_count += len(object_segment_one_line_lane_list)

        # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
        # 并将初始化后的对象存入total_images_data_list
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
            process_output['temp_file_name_list'].append(image.file_name_new)
            process_output['success_count'] += 1
        else:
            print('errow output temp annotation: {}'.format(image.file_name_new))
            process_output['fail_count'] += 1

        return

    @staticmethod
    def target_dataset(dataset_instance: Dataset_Base):
        """[输出temp dataset annotation]

        Args:
            dataset (Dataset): [dataset]
        """

        print('\nStart transform to target dataset:')

        return

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取BDD100K数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        return

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成COCO 2017组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')

        return
