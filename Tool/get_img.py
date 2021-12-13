'''
Description: 
Version: 
Author: Leidi
Date: 2021-12-08 16:34:03
LastEditors: Leidi
LastEditTime: 2021-12-08 16:47:40
'''
from tqdm import tqdm
import cv_bridge
import roslib
import rosbag
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

image_output_path = r'/mnt/data_1/Data/rosbag/images/qunguang_2021-12-07-11-47-22'
bag_path = r'/mnt/data_1/Data/rosbag/2021-12-07-11-47-22.bag'

class ImageCreate():
    def __init__(self):
        self.bridge = CvBridge()
        i = 0
        num = 0
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in tqdm(bag.read_messages()):
                # print(i)
                if topic == "/cam_left_front/csi_cam/image_raw/compressed":
                    i += 1
                    if i % 3 != 0:
                        continue
                    try:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(
                            msg, "bgr8")
                    except CvBridgeError as e:
                        print(e)
                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    image_name = timestr+".png"
                    # if not os.exist(path):
                    #     os.makedirs(path)
                    cv2.imwrite(image_output_path+'cam_left_front/'+image_name, cv_image)
                    # time.sleep(0.9)
                if topic == "/cam_front_center/csi_cam/image_raw/compressed":
                    num += 1
                    if num % 3 != 0:
                        continue
                    try:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(
                            msg, "bgr8")
                    except CvBridgeError as e:
                        print(e)
                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    image_name = timestr+".png"
                    # if not os.exist(path):
                    #     os.makedirs(path)
                    cv2.imwrite(image_output_path+'cam_front_center/'+image_name, cv_image)
                if topic == "/cam_right_front/csi_cam/image_raw/compressed":
                    num += 1
                    if num % 3 != 0:
                        continue
                    try:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(
                            msg, "bgr8")
                    except CvBridgeError as e:
                        print(e)
                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    image_name = timestr+".png"
                    # if not os.exist(path):
                    #     os.makedirs(path)
                    cv2.imwrite(image_output_path+'cam_front_center/'+image_name, cv_image)
                    # time.sleep(0.9)

                    # time.sleep(0.9)
            # print("=================",num)
if __name__ == '__main__':
    try:
        image_creator = ImageCreate()
    except rospy.ROSInterruptException:
        pass
