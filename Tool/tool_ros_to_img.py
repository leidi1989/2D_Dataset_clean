'''
Description: 
Version: 
Author: Leidi
Date: 2021-12-08 16:34:03
LastEditors: Leidi
LastEditTime: 2021-12-22 17:17:51
'''
import cv2
import rospy
import rosbag
import argparse
from tqdm import tqdm
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError


image_output_path = r'/mnt/data_1/Data/rosbag/images/qunguang_2021-12-07-11-47-22'
bag_path = r'/mnt/data_1/Data/rosbag/qunguang/20211209/1209_qunguang1_2_2021-12-09-11-01-17_1.bag'


class ImageCreate():
    def __init__(self):
        self.bridge = CvBridge()
        i = 0
        num = 0
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in tqdm(bag.read_messages()):
                if topic == '/cam_left_front/csi_cam/image_raw/compressed':
                    i += 1
                    if i % 3 != 0:
                        continue
                    try:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(
                            msg, 'bgr8')
                    except CvBridgeError as e:
                        print(e)
                    timestr = '%.6f' % msg.header.stamp.to_sec()
                    image_name = timestr+'.png'
                    cv2.imwrite(image_output_path +
                                'cam_left_front/'+image_name, cv_image)
                if topic == '/cam_front_center/csi_cam/image_raw/compressed':
                    num += 1
                    if num % 3 != 0:
                        continue
                    try:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(
                            msg, 'bgr8')
                    except CvBridgeError as e:
                        print(e)
                    timestr = '%.6f' % msg.header.stamp.to_sec()
                    image_name = timestr+'.png'
                    cv2.imwrite(image_output_path +
                                'cam_front_center/'+image_name, cv_image)
                if topic == '/cam_right_front/csi_cam/image_raw/compressed':
                    num += 1
                    if num % 3 != 0:
                        continue
                    try:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(
                            msg, 'bgr8')
                    except CvBridgeError as e:
                        print(e)
                    timestr = '%.6f' % msg.header.stamp.to_sec()
                    image_name = timestr+'.png'
                    cv2.imwrite(image_output_path +
                                'cam_front_center/'+image_name, cv_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='tool_avi_to_mp4.py')
    parser.add_argument('--avipath', default=r'',
                        type=str, help='avi path')
    parser.add_argument('--imgpath', default=r'/mnt/data_1/Dataset/detect_output/YOLOP/20211209/cam_front_center_20210918_output_20211209',
                        type=str, help='image output path')
    parser.add_argument('--pref', default=r'',
                        type=str, help='rename prefix')
    parser.add_argument('--mode', default='',
                        type=str, help='image output')
    parser.add_argument('--time', default=1,
                        type=int, help='the time of create image, secend')
    parser.add_argument('--mp4fps', default=30,
                        type=int, help='the fps of concate images.')
    opt = parser.parse_args()

    try:
        image_creator = ImageCreate()
    except rospy.ROSInterruptException as e:
        print(e)
