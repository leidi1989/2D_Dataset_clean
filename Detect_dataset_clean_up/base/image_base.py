'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:13:19
LastEditors: Leidi
LastEditTime: 2021-11-08 16:48:24
'''
import os
from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms


class TRUE_BOX:
    """真实框类"""

    def __init__(self,
                 clss: str,
                 xmin: float,
                 ymin: float,
                 xmax: float,
                 ymax: float,
                 color: str = '',
                 tool: str = '',
                 difficult: int = 0,
                 distance: float = 0,
                 occlusion: float = 0
                 ) -> None:
        """[真实框类]

        Args:
            clss (str): [description]
            xmin (float): [description]
            ymin (float): [description]
            xmax (float): [description]
            ymax (float): [description]
            color (str, optional): [description]. Defaults to ''.
            tool (str, optional): [description]. Defaults to ''.
            difficult (int, optional): [description]. Defaults to 0.
            distance (float, optional): [description]. Defaults to 0.
            occlusion (float, optional): [description]. Defaults to 0.
        """
        self.clss = clss
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.color = color              # 真实框目标颜色
        self.tool = tool                # bbox工具
        self.difficult = difficult      # 困难样本
        self.distance = distance        # 真实框中心点距离
        self.occlusion = occlusion      # 真实框遮挡率


class IMAGE:
    """图片类"""

    def __init__(self,
                 image_name_in: str,
                 image_name_new_in: str,
                 image_path_in: str,
                 height_in: int,
                 width_in: int,
                 channels_in: int,
                 true_box_list_in: list) -> None:
        """[图片类]

        Args:
            image_name_in (str): [description]
            image_name_new_in (str): [description]
            image_path_in (str): [description]
            height_in (int): [description]
            width_in (int): [description]
            channels_in (int): [description]
            true_box_list_in (list): [description]
        """
        self.image_name = image_name_in                     # 图片名称
        self.image_name_new = image_name_new_in             # 修改后图片名称
        self.file_name = os.path.splitext(self.image_name)[0]
        self.file_name_new = os.path.splitext(self.image_name_new)[0]
        self.image_path = image_path_in                     # 图片地址
        self.height = height_in                             # 图片高
        self.width = width_in                               # 图片宽
        self.channels = channels_in                         # 图片通道数
        self.true_box_list = true_box_list_in               # 图片真实框列表

    def true_box_list_updata(self, one_bbox_data: TRUE_BOX) -> None:
        """[为per_image对象true_box_list成员添加元素]

        Args:
            one_bbox_data (true_box): [TRUE_BOX类真实框变量]
        """

        self.true_box_list.append(one_bbox_data)


# class MyDataset(Dataset):
#     """[读取自定义数据集]

#     Args:
#         Dataset ([type]): [touch读取数据集基类]
#     """    
#     def __init__(self, txt_path:str):
#         """[读取自定义数据集初始化]

#         Args:
#             txt_path (str): [图片路径列表]
#         """       
#         fh = open(txt_path, 'r')
#         imgs = []
#         for line in fh:
#             line = line.rstrip()
#             words = line.split()
#             imgs.append(words[0])
#         self.imgs = imgs

#     def __getitem__(self, index):
#         fn = self.imgs[index]
#         img = Image.open(fn).convert('RGB')
#         transf = transforms.ToTensor()
#         img_tensor = transf(img)
#         return img_tensor, fn

#     def __len__(self):
#         return len(self.imgs)
