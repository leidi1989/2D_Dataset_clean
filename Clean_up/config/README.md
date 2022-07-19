<!--
 * @Description: 
 * @Version: 
 * @Author: Leidi
 * @Date: 2022-07-19 14:53:07
 * @LastEditors: Leidi
 * @LastEditTime: 2022-07-19 15:35:21
-->
> Dataset_input_folder: (/path)

输入数据集路径

> Dataset_output_folder: (/path)

输出数据集路径

> Source_dataset_style: (YUNCE_SEGMENT_COCO or ...)

源数据集输入类型：

[YUNCE_SEGMENT_COCO, YUNCE_SEGMENT_COCO_ONE_IMAGE, HUAWEIYUN_SEGMENT, CVAT_COCO2017, HY_VAL, SJT, MYXB, HY_HIGHWAY, GEELY_COCO_ONE_IMAGE YOLO, COCO2017, BDD100K, TT100K, CCTSDB, LISA, KITTI, APOLLOSCAPE_LANE_SEGMENT, TI_EDGEAILITE_AUTO_ANNOTATION]

> Target_dataset_style: (COCO2017 or ...)

输入目标数据集输入类型
[COCO2017, CITYSCAPES, CITYSCAPES_VAL, CVAT_IMAGE_1_1, YOLO, CROSS_VIEW]

> Only_static: (True or False)

是否只进行统计

> File_prefix: (any str)

图片及标注文件添加的前缀

> File_prefix_delimiter: "@"

图片及标注文件添加的前缀分隔符(cityscapes need: "@")

> Target_dataset_divide_proportion: 0.8,0.1,0.1,0

训练集，验证集，测试集，冗余数据分配比例(train, val, test, redund)

> Target_dataset_check_annotations_count: （100）

绘制输出目标数据集标签检测图片数量

> Target_dataset_check_annotations_output_as_mask: True

是否使用透明掩码绘制输出目标数据集标签检测图片

> Need_convert: （segmentation_to_box or box_to_segmentation）

是否进行标签类型转化，由分割转换为box，或者由box转换为分割（segmentation_to_box, box_to_segmentation）

> Label_object_rotation_angle: 0

标注框旋转角度

> Task_and_class_config:  
   Detection:  
     Source_dataset_class_file_path:(/.name file path)  
     Modify_class_file_path:(/.txt file path)  
     Target_each_class_object_pixel_limit_file_path:(/limit .txt path)  
   Semantic_segmentation:  
    Source_dataset_class_file_path:(/.name file path)  
     Modify_class_file_path:(/.txt file path)  
     Target_each_class_object_pixel_limit_file_path:(/limit .txt path)  
   Instance_segmentation:  
     Source_dataset_class_file_path:(/.name file path)  
     Modify_class_file_path:(/.txt file path)  
     Target_each_class_object_pixel_limit_file_path:(/limit .txt path)  
   Keypoints:  
     Source_dataset_class_file_path:(/path)  
     Modify_class_file_path:(/path)  
     Target_each_class_object_pixel_limit_file_path:(/path)  

任务和任务类别配置，需选择一项填写（Detection, Semantic_segmentation, Instance_segmentation, Keypoints），并给出对应的类别文件路径，类别修改文件路径，类别距离挑选文件路径

# debug: （True or False）
debug模式
