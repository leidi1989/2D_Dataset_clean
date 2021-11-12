<!--
 * @Description: 
 * @Version: 
 * @Author: Leidi
 * @Date: 2021-04-26 20:59:02
 * @LastEditors: Leidi
 * @LastEditTime: 2021-05-15 17:53:21
-->
# Dataset clean

### 背景介绍
将不同2D数据集转换为YOLO系列组织形式数据集。

### 功能说明
1. 将原有文件夹图片转移至images并对图片添加前缀。
2. 将原有文件夹中的标签转移至source label，并更换标签文件中图片名称。
3. 依据标签融合字典txt文件，将不同数据集标签类别信息统一。
4. 将source label中的bbox信息转换为Pascal格式的xml文件保存至Annotations。
5. 按设置比例生成train、val、test、redund集，其中redund集为留置。
6. 生成train、val、test、redund集分布图及统计分析。
7. 绘制真实框以供检查label正确性。

### 性能说明
将原始数据集整理为以下组织结构：
![alt](.\ldp组织图.png)

### 方案说明
将源图片、标签按YOLO格式进行整理。

### 运行环境
```python
import argparse
import os
import shutil
import time
from utils.utils import *
from utils.extract_function import *
import cleaning_1_images
import cleaning_2_keys
import cleaning_3_extract
import cleaning_4_annotations_integrated
import cleaning_5_labels
import cleaning_6_divideset
import cleaning_7_distribution
import cleaning_8_checklabels
```

### 接口说明
```python
--set               # 待整理数据集路径
--names             # 数据集类别文件路径（classes.names）
--out               # 整理后数据集输出路径
--pref              # 图片添加前缀
--ilstyle           # 输出数据集类型:
                    #（ldp, sjt, hy, hy_highway,pascal, kitti,
                    # coco, lisa, hanhe，sjt）
--olstyle           # 输出数据集类型
--fixname           # 是否融合标签
--fixnamesfile      # 融合标签字典路径
--mod               # 输出label类型（yolo, yolo_2）
--ratio             # 输出train、val、test、redund集比例
                    # （train, val, test, redund）
--check             # 是否检查bbox正确性，及在图片中绘制bbox
--mask              # 是否采取透明框形式绘制bbox
```

### 软件架构
程序设计思维导图如下：
![alt](.\数据集程序流程图.png)

### 安装教程
无需安装，下载即用。

### 使用说明
```cmd
python cleaning.py --set=E:\2.Datasets\KITTI_data_input --names=E:\2.Datasets\KITTI_data_input\classes.names --out=E:\2.Datasets\KITTI_data_input_analyze_mix --pref=kitti --ilstyle=kitti --olstyle=ldp --fixname=1 --fixnamesfile=D:\dataset\dataset_labels_integrated_dict\total_mix_20201207.txt --mod=yolo --ratio=0.5, 0.25, 0.25, 0 --check=1 --mask=1
```
其中，各参数解释如下：
```python
--set               # 待整理数据集路径
--names             # 数据集类别文件路径（classes.names）
--out               # 整理后数据集输出路径
--pref              # 图片添加前缀
--ilstyle           # 输出数据集类型:
                    #（ldp, sjt, hy, hy_highway,pascal, kitti,
                    # coco, lisa, hanhe，sjt）
--olstyle           # 输出数据集类型
--fixname           # 是否融合标签
--fixnamesfile      # 融合标签字典路径
--mod               # 输出label类型（yolo, yolo_2）
--ratio             # 输出train、val、test、redund集比例
                    # （train, val, test, redund）
--check             # 是否检查bbox正确性，及在图片中绘制bbox
--mask              # 是否采取透明框形式绘制bbox
```

### 更新说明
暂无