'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:03
LastEditors: Leidi
LastEditTime: 2021-05-15 17:56:06
'''
import json as js
import cv2
import os
import numpy as np
import random
from tqdm import tqdm


values = ['car tail', 'car full', 'pedestrian',
          'motorcycle', 'bicycle', 'cyclist', 'truck tail',
          'truck full', 'bus tail', 'bus full', 'tricycle tail',
          'tricycle full', 'van tail', 'van full', 'special vehicle tail',
          'special vehicle full', 'engineering vehicle tail',
          'engineering vehicle full', 'pushcart', 'other vehicle tail', 'other vehicle full',
          'cone barrel', 'road stake', 'passage pole', 'trash', 'water barriers', 'billboard', 'finger plate',
          "traffic light", "Danger signs", "speed limit", "no parking", "parking lot", "motorway", "bicycle lane",
          "no entry", "pedestrian crossing", "warning sign", "polygon"]
# 不同类别的框用不同颜色区分
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(38)]
# 统计各个类别的框数
nums = [[] for _ in range(39)]

label_num = 1
# package = "highway/night-time"
# total_path = os.path.abspath('.')
# image_path = os.path.join(total_path + "/" + package + "/" + str(label_num))
# label_path = os.path.join(total_path + "/" + package +
#                           "/" + str(label_num) + "-label")
# dirs = os.listdir(label_path)

package = r'D:\DataSets\cdx\cdx\urban\sunny-time\3_input'
savepath = r'D:\DataSets\myxb_df_highway_urban_input_analyze\count_box'
pic_path = r'D:\DataSets\myxb_df_highway_urban_input_analyze\images'
label_path = r'D:\DataSets\myxb_df_highway_urban_input_analyze\source_label'
total_path = os.path.abspath('.')
without_lane_images = []
# json_list = os.listdir(json_path)
dirs = os.listdir(label_path)


total_box_count = 0
for json_path in tqdm(dirs):
    json_path = os.path.join(label_path, json_path)
    # savepath = os.path.join(total_path + "/" + package +
                            # "/" + str(label_num) + "-show")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # print(json_path)
    with open(json_path, 'r') as f:
        data = js.load(f)

        # data.sort(key = lambda x: int(x['imageName'].split('.')[0])) ##文件名按数字排序
        for d in data:
            img = cv2.imread(os.path.join(pic_path, d['imageName']))
            if d["Data"] == []:
                break
            for box in d["Data"]["svgArr"]:
                # 画车线
                if box["tool"] == 'polygon':
                    pointx = []
                    pointy = []
                    cls = 'polygon'
                    nums[values.index(cls)].append(cls)
                    for point in box['data']:
                        pointx.append(point['x'])
                        pointy.append(point['y'])
                    points = np.array(list(zip(pointx, pointy)))
                    cv2.polylines(img, np.int32([points]), 1, (0, 0, 255), 1)
                    total_box_count += 1
                else:
                    # 画2D框
                    xmin = int(box['data'][0]['x'])
                    ymin = int(box['data'][0]['y'])
                    xmax = int(box['data'][2]['x'])
                    ymax = int(box['data'][2]['y'])
                    cls = box['secondaryLabel'][0]['value']
                    try:
                        num = nums[values.index(cls)].append(cls)
                        color = colors[values.index(cls)]
                        cv2.rectangle(img, (xmin, ymin),
                                      (xmax, ymax), color, 2)
                        cv2.putText(img, cls, (xmin, ymin),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
                        total_box_count += 1
                    except ValueError:
                        print(d['imageName'] + " ValueError: " +
                              cls + "is not in list")
                        pass

            path = os.path.join(savepath, d['imageName'])

            cv2.imwrite(path, img)

for i in nums:
    if len(i) != 0:
        print(i[0] + ':' + str(len(i)))

print(total_box_count)
with open(r'D:\DataSets\myxb_df_highway_urban_input_analyze\ImageSets\count_box.txt', 'w') as f:
    f.write(str(total_box_count) + '\n')

print('Done!')
