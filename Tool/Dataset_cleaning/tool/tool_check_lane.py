'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:03
LastEditors: Leidi
LastEditTime: 2021-12-21 16:15:50
'''
# -*- coding: utf-8 -*-
import os
import sys
import json
import cv2
from tqdm import tqdm
import numpy as np


label_num = 3

# TuSimple数据集格式中采样点
y_axis_list = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390,
               400, 410, 420, 430, 440, 450, 460, 470, 480, 490,
               500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
               600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]

# 这里使用10种颜色去填充mask，应该车道线不会超过16条吧
fill_mask_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (
    0, 255, 255), (255, 255, 255), (125, 125, 125), (0, 0, 125), (0, 125, 125), (125, 0, 0), (
        0, 134, 139), (64, 224, 208), (160, 82, 45), (255, 192, 203), (205, 190, 112), (121, 231, 34), (34, 56, 112), (75, 97, 112), (46, 38, 156)]


def get_base_name(full_name):
    base_name = os.path.basename(full_name)[:-5]
    return base_name


'''
定义：平面上的三点P1(x1,y1),P2(x2,y2),P3(x3,y3)
    S(P1,P2,P3)= (x1-x3)* (y2-y3) - (y1-y3)*(x2-x3)
    令矢量的起点为A，终点为B，判断的点为C，
    如果S（A，B，C）为正数，则C在矢量AB的左侧；
    如果S（A，B，C）为负数，则C在矢量AB的右侧；
    如果S（A，B，C）为 0，则C在直线AB上。
'''
def calculate_s(left_point, right_point, point):
    s = (left_point[0] - point[0])*(right_point[1] - point[1]) - \
        (left_point[1] - point[1]) * \
        (right_point[0] - point[0])
    return s


'''
遍历图片中的车道线信息
'''
def get_lane_label(svgArr):
    lane_box = [[]]
    lane_categaries = []
    lane_flag = []
    lane_box_num = 0
    index = 0
    for item in svgArr:
        # 只针对车道线标注
        if(item['name'] == "lane line"):
            lane_categary = item['secondaryLabel'][0]['value']
            lane = []
            lane_box_num += 1
            points = item['data']
            for point in points:
                point_x = int(point['x'])
                point_y = int(point['y'])
                lane.append([point_x, point_y])

            if len(lane_categaries) == 0:
                lane_categaries.append(lane_categary)
                if("dotted" in lane_categary):
                    lane_flag.append(0)
                else:
                    lane_flag.append(1)
                for point in lane:
                    lane_box[0].append(point)
                index = 1

            elif lane_categary not in lane_categaries:
                lane_categaries.append(lane_categary)
                if("dotted" in lane_categary):
                    lane_flag.append(0)
                else:
                    lane_flag.append(1)
                lane_box.append([])
                for point in lane:
                    lane_box[index].append(point)
                index += 1
            else:
                find_index = lane_categaries.index(lane_categary)
                for point in lane:
                    lane_box[find_index].append(point)
    return lane_flag, lane_box


'''
将车道线信息生成mask
'''
def generate_mask(img, lane_box, lane_flag):
    flag_index = 0
    img_mask = np.zeros(img.shape, dtype="uint8")
    for lane_point in lane_box:
        lane_point = np.array(lane_point)
        x, y = np.split(lane_point, 2, axis=1)
        x = x.flatten()
        y = y.flatten()

        if lane_flag[flag_index] == 1:
            cv2.fillPoly(img, [lane_point], 0, 1)
            cv2.fillPoly(img_mask, [lane_point],
                         fill_mask_color[flag_index], 1)
        else:
            # 所有点按照x从小到大排序
            ind = np.argsort(x, axis=0)
            out_x = np.take_along_axis(x, ind[::], axis=0).tolist()
            out_y = np.take_along_axis(y, ind[::], axis=0).tolist()

            # 找到x值最小的点，称为left_point，找到x值最大的点称为right_point
            left_point = [out_x.pop(0), out_y.pop(0)]
            right_point = [out_x.pop(), out_y.pop()]

            up_points = []
            down_points = []
            output_points = []
            # 将left_point 和right_point连成一条线，这条线将剩余的点分为两部分，上部分和下部分
            other_points = np.column_stack((out_x, out_y))
            for point in other_points:
                # 判断点在直线的上方还是下方
                s = calculate_s(left_point, right_point, point)
                if(s > 0):
                    # 点在直线上方
                    up_points.append(point)
                else:
                    # 点在直线下方
                    down_points.append(point)
            # 将直线上方的点，按照x从小到大排序,因为已经排序过了，所以不需要重新排序
            output_points.append(np.array(left_point))
            for up_point in up_points:
                output_points.append(up_point)
            output_points.append(np.array(right_point))
            # 将直线下方的点，按照x从大到小顺序排序，加入到输出中
            down_points_num = len(down_points)
            for i in range(down_points_num):
                output_points.append(
                    np.array(down_points[down_points_num - i - 1]))

            cv2.fillPoly(img, np.array([output_points]), 0, 1)
            cv2.fillPoly(img_mask, np.array([output_points]),
                         fill_mask_color[flag_index], 1)

        flag_index += 1
        return img_mask


'''
从mask图像中得到线的中心点
'''
def get_center_point(img_mask, img):
    mask = img_mask.copy()
    height = mask.shape[0]
    wight = mask.shape[1]
    channels = mask.shape[2]

    x_tmp = []
    zore_point = np.array([0, 0, 0])
    last_point = np.array([0, 0, 0])
    for row in y_axis_list:  # 遍历高
        for col in range(wight):  # 遍历宽
            pix = mask[row][col]
            if not (pix == last_point).all():
                # print("last_point: {}".format(last_point))
                last_point = pix
                x_tmp.append(col)

        if(len(x_tmp) % 2 == 1):
            x_tmp.append(wight)

        for i in range(int(len(x_tmp)/2)):
            center_x = int((x_tmp[2*i] + x_tmp[2*i+1]) / 2)
            cv2.circle(img_mask, (center_x, row), 10, fill_mask_color[i], 1)
            cv2.circle(img, (center_x, row), 10, fill_mask_color[i], 1)
            # cv2.imshow("1", img_mask)
            # cv2.waitKey(0)
        x_tmp = []
        last_point = np.array([0, 0, 0])

    # cv2.imshow("1", img_mask)
    # cv2.waitKey(0)


def main():
    
    package = r'D:\DataSets\cdx\cdx\urban\sunny-time\3_input'
    savepath = r'D:\DataSets\cdx\cdx\urban\sunny-time\3_input_lane'
    pic_path = os.path.join(package, 'images')
    json_path = os.path.join(package ,'labels')
    total_path = os.path.abspath('.')
    without_lane_images = []
    json_list = os.listdir(json_path)
    # for josn_file in glob.glob(json_path):
    print('From %s', package)
    lane_box_count = 0
    for josn_file in tqdm(json_list):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        print(josn_file)
        with open(os.path.join(json_path, josn_file), 'r') as f:
            # base_name = get_base_name(josn_file)
            content = f.read()
            if content.startswith(u'\ufeff'):
                content = content.encode('utf8')[3:].decode('utf8')
            infos = json.loads(content)
            pic_num = len(infos)
            # print("共有 {} 张图片标注信息: ".format(pic_num))
            # print("***************************")

            # 遍历所有图片
            # infos.sort(key = lambda x: int(os.path.splitext(x['imageName'])[0])) ##文件名按数字排序
            for info in tqdm(infos):
                a1 = info['Data']
                image_name = info['imageName']
                image_file = os.path.join(pic_path, image_name)
                # print("image: ", image_file)
                if(len(a1) == 0):
                    without_lane_images.append(image_name)
                    continue
                img = cv2.imread(image_file)
                # src_img = img.copy()
                img_mask = np.zeros(img.shape, dtype="uint8")
                if img is None:
                    # print("load img error")
                    sys.exit()

                svgArr = a1["svgArr"]

                lane_box = [[]]
                lane_flag = []  # 用来标记实线和虚线，1代表实线，0代表虚线

                lane_flag, lane_box = get_lane_label(svgArr)

                lane_num = len(lane_box)
                if(len(lane_box[0]) == 0):
                    # print("此图片共有 0 条车道线: ")
                    continue
                # print("此图片共有{}条车道线: ".format(lane_num))
                # lane_box_count += lane_num

                flag_index = 0
                # img_mask = generate_mask(img,lane_box,lane_flag)
                for lane_point in lane_box:
                    lane_point = np.array(lane_point)
                    x, y = np.split(lane_point, 2, axis=1)
                    x = x.flatten()
                    y = y.flatten()

                    if lane_flag[flag_index] == 1:
                        cv2.fillPoly(img, [lane_point],
                                        fill_mask_color[flag_index], 1)
                        cv2.fillPoly(img_mask, [lane_point],
                                        fill_mask_color[flag_index], 1)
                    else:
                        # 所有点按照x从小到大排序
                        ind = np.argsort(x, axis=0)
                        out_x = np.take_along_axis(x, ind[::], axis=0).tolist()
                        out_y = np.take_along_axis(y, ind[::], axis=0).tolist()

                        # 找到x值最小的点，称为left_point，找到x值最大的点称为right_point
                        left_point = [out_x.pop(0), out_y.pop(0)]
                        right_point = [out_x.pop(), out_y.pop()]

                        up_points = []
                        down_points = []
                        output_points = []
                        # 将left_point 和right_point连成一条线，这条线将剩余的点分为两部分，上部分和下部分
                        other_points = np.column_stack((out_x, out_y))
                        for point in other_points:
                            # 判断点在直线的上方还是下方
                            s = calculate_s(left_point, right_point, point)
                            if(s > 0):
                                # 点在直线上方
                                up_points.append(point)
                            else:
                                # 点在直线下方
                                down_points.append(point)
                        # 将直线上方的点，按照x从小到大排序,因为已经排序过了，所以不需要重新排序
                        output_points.append(np.array(left_point))
                        for up_point in up_points:
                            output_points.append(up_point)
                        output_points.append(np.array(right_point))
                        # 将直线下方的点，按照x从大到小顺序排序，加入到输出中
                        down_points_num = len(down_points)
                        for i in range(down_points_num):
                            output_points.append(
                                np.array(down_points[down_points_num - i - 1]))

                        cv2.fillPoly(img, np.array(
                            [output_points]), fill_mask_color[flag_index], 1)
                        # cv2.fillPoly(img_mask, np.array([output_points]),
                        #              fill_mask_color[flag_index], 1)

                    flag_index += 1

                # get_center_point(img_mask,src_img)

                path = os.path.join(savepath, info['imageName'])

                cv2.imwrite(path, img)

                # cv2.namedWindow("1", cv2.WINDOW_NORMAL)
                # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
                # cv2.namedWindow("src", cv2.WINDOW_NORMAL)
                # cv2.imshow("1", img)
                # cv2.imshow("mask", img_mask)
                # cv2.imshow("src", src_img)
                # cv2.waitKey(0)
                # print("****************")
    print('Done, save to %s', savepath)


if __name__ == "__main__":
    main()


'''
车道线点，多项式拟合
for t in lane_box:
    # 将车道线点的x和y分离
    t = np.array(t)
    x , y = np.split(t,2,axis=1)
    x = x.flatten()
    y = y.flatten()
    for i in range(len(x)):
        cv2.circle(img,(x[i],y[i]),5,(0,255,0))
    # 使用多项式拟合这些点，poly为多项式参数
    poly = np.polyfit(x,y,2)
    # fx为多项式函数
    fx = np.poly1d(poly)
    print(fx)
    Y = fx(x)
    Y = np.int32(Y)
    # 将拟合出来多项式的点在图上画出来
    poly_points = np.column_stack((x,Y))
    for point in poly_points:
        cv2.circle(img,(point[0],point[1]),5,(0,0,255))
    cv2.polylines(img, [poly_points], isClosed=False, color=(255,255,255), thickness=1)
    '''


'''
直接找点的凸包就可以
但是对于非凸的形状不work

hull = cv2.convexHull(lane_point)
print("hull: ",hull)

cv2.drawContours(img,[hull],0,(0,0,255),2)
'''
