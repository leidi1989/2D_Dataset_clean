'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-17 16:44:24
'''
from utils.utils import *
from base.image_base import *
from base.dataset_base import Dataset_Base


class SJT(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['png']
        self.source_dataset_annotation_form = 'txt'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count()

    def load_image_annotation(self, source_annotation_name: str, process_output: dict) -> None:
        """将源标注转换为暂存标注

        Args:
            source_annotation_name (str): 源标注文件名称
            process_output (dict): 进程间通信字典
        """

        source_annotation_path = os.path.join(
            self.source_dataset_annotations_folder, source_annotation_name)
        image_name = os.path.splitext(source_annotation_name)[
            0] + '.' + self.temp_image_form
        image_name_new = self.file_prefix + image_name
        image_path = os.path.join(
            self.temp_images_folder, image_name_new)
        img = cv2.imread(image_path)
        if img is None:
            print('Can not load: {}'.format(image_name_new))
            return
        height, width, channels = img.shape     # 读取每张图片的shape
        with open(source_annotation_path, 'r') as f:
            data = json.load(f)
        object_list = []
        if len(data['boxs']):
            for one_box in data['boxs']:
                # 读取json文件中的每个真实框的class、xy信息
                x = one_box['x']
                y = one_box['y']
                xmin = min(max(float(x), 0.), float(width))
                ymin = min(max(float(y), 0.), float(height))
                xmax = max(min(float(x + one_box['w']), float(width)), 0.)
                ymax = max(min(float(y + one_box['h']), float(height)), 0.)
                true_box_color = ''
                cls = ''
                types = one_box['Object_type']
                types = types.replace(' ', '').lower()
                if types == 'pedestrians' or types == 'vehicles' or types == 'trafficlights':
                    cls = one_box['Category']
                    if types == 'trafficlights':    # 获取交通灯颜色
                        true_box_color = one_box['Color']
                        true_box_color = true_box_color.replace(
                            ' ', '').lower()
                else:
                    cls = types
                cls = cls.replace(' ', '').lower()
                if cls == 'misc' or cls == 'dontcare':
                    continue
                ture_box_occlusion = 0
                if 'Occlusion' in one_box:
                    ture_box_occlusion = self.change_Occlusion(
                        one_box['Occlusion'])  # 获取真实框遮挡率
                ture_box_distance = 0
                if 'distance' in one_box:
                    ture_box_distance = one_box['distance']  # 获取真实框中心点距离
                if xmax > xmin and ymax > ymin:
                    # 将单个真实框加入单张图片真实框列表
                    box_xywh = [int(xmin), int(ymin), int(
                        xmax-xmin), int(ymax-ymin)]
                    object_list.append(OBJECT(0,
                                                cls,
                                                box_clss=cls,
                                                box_xywh=box_xywh,
                                                box_occlusion=ture_box_occlusion,
                                                box_distance=ture_box_distance))
                else:
                    print('\nBbox error!')
                    continue
                box_xywh = [int(xmin), int(ymin), int(
                    xmax-xmin), int(ymax-ymin)]
                object_list.append(OBJECT(0,
                                            cls,
                                            box_clss=cls,
                                            box_xywh=box_xywh,
                                            box_occlusion=ture_box_occlusion,
                                            box_distance=ture_box_distance))
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
            process_output['temp_file_name_list'].append(
                image.file_name_new)
            process_output['success_count'] += 1
        else:
            print('errow output temp annotation: {}'.format(
                image.file_name_new))
            process_output['fail_count'] += 1

        return

    def change_traffic_light(self, true_box_dict_list: list) -> list:
        """[修改数据堂信号灯标签信息, 将灯与信号灯框结合]

        Args:
            true_box_dict_list (list): [源真实框]

        Returns:
            list: [修改后真实框]
        """

        light_name = ['ordinarylight',
                      'goingstraight',
                      'turningleft',
                      'turningright',
                      'u-turn',
                      'u-turn&turningleft',
                      'turningleft&goingstraight',
                      'turningright&goingstraight',
                      'u-turn&goingstraight',
                      'numbers']
        light_base_name = ['trafficlightframe_'+i for i in light_name]
        light_base_name.append('trafficlightframe')
        light_go = []   # 声明绿灯信号灯命名列表
        light_stop = []     # 声明红灯信号灯命名列表
        light_warning = []      # 声明黄灯信号灯命名列表
        for one_name in light_base_name:
            light_go.append(one_name + '_' + 'green')
        for one_name in light_base_name:
            light_stop.append(one_name + '_' + 'red')
        for one_name in light_base_name:
            light_stop.append(one_name + '_' + 'yellow')
            light_stop.append(one_name + '_' + 'unclear')
            light_stop.append(one_name + '_' + 'no')
        light_numbers = []
        light_numbers.append('trafficlightframe_numbers' + '_' + 'green')
        light_numbers.append('trafficlightframe_numbers' + '_' + 'red')
        light_numbers.append('trafficlightframe_numbers' + '_' + 'yellow')
        light_numbers.append('trafficlightframe_numbers' + '_' + 'unclear')
        light_numbers.append('trafficlightframe_numbers' + '_' + 'no')
        new_true_box_dict_list = []  # 声明新真实框列表
        for one_true_box in true_box_dict_list:  # 遍历源真实框列表
            if one_true_box.clss == 'trafficlightframe':    # 搜索trafficlightframe真实框
                if one_true_box.color == 'no':
                    for light_true_box in true_box_dict_list:    # 遍历源真实框列表
                        if (light_true_box.clss in light_name and
                            light_true_box.xmin >= one_true_box.xmin - 20 and
                            light_true_box.ymin >= one_true_box.ymin - 20 and
                            light_true_box.xmax <= one_true_box.xmax + 20 and
                                light_true_box.ymax <= one_true_box.ymax + 20):  # 判断信号灯框类别
                            one_true_box.clss += (
                                '_' + light_true_box.clss + '_' + light_true_box.color)   # 新建信号灯真实框实例并更名
                        if one_true_box.clss in light_numbers:
                            continue
                        if one_true_box.clss in light_go:     # 将信号灯归类
                            one_true_box.clss = 'go'
                        if one_true_box.clss in light_stop:
                            one_true_box.clss = 'stop'
                        if one_true_box.clss in light_warning:
                            one_true_box.clss = 'warning'
                    if one_true_box.clss == 'trafficlightframe':    # 若为发现框内有信号灯颜色则更换为warning
                        one_true_box.clss = 'warning'
                else:
                    innate_light = ''
                    for light_true_box in true_box_dict_list:    # 遍历源真实框列表
                        if (light_true_box.clss in light_name and
                            light_true_box.xmin >= one_true_box.xmin - 20 and
                            light_true_box.ymin >= one_true_box.ymin - 20 and
                            light_true_box.xmax <= one_true_box.xmax + 20 and
                                light_true_box.ymax <= one_true_box.ymax + 20):  # 判断信号灯框类别
                            innate_light = light_true_box.clss
                    if innate_light == 'numbers':
                        continue
                    one_true_box.clss += ('_' + one_true_box.color)
                    if one_true_box.clss in light_go:     # 将信号灯归类
                        one_true_box.clss = 'go'
                    if one_true_box.clss in light_stop:
                        one_true_box.clss = 'stop'
                    if one_true_box.clss in light_warning:
                        one_true_box.clss = 'warning'
            # if one_true_box.clss == 'numbers':
            #     if one_true_box.color == 'green':     # 将数字类信号灯归类
            #         one_true_box.clss = 'go'
            #     if one_true_box.color == 'red':
            #         one_true_box.clss = 'stop'
            #     if (one_true_box.color == 'yellow' or
            #         one_true_box.color == 'no' or
            #         one_true_box.color == 'unclear'):
            #         one_true_box.clss = 'warning'
            new_true_box_dict_list.append(one_true_box)

        return new_true_box_dict_list

    def change_Occlusion(self, source_occlusion: str) -> int:
        """[转换真实框遮挡信息]

        Args:
            source_occlusion (str): [ture box遮挡信息]

        Returns:
            int: [返回遮挡值]
        """

        occlusion = 0
        if source_occlusion == "No occlusion (0%)":
            occlusion = 0
        if source_occlusion == "Partial occlusion (0%~35%)":
            occlusion = 35
        if source_occlusion == "Occlusion for most parts (35%~50%)":
            occlusion = 50
        if source_occlusion == "Others (more than 50%)":
            occlusion = 75

        return occlusion

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
        """[读取SJT数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成SJT组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')

        return
