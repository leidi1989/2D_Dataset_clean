'''
Description: 
Version: 
Author: Leidi
Date: 2021-12-22 18:21:36
LastEditors: Leidi
LastEditTime: 2021-12-22 18:35:18
'''
import cv2
import re
import os
import os.path
import rospy
import rosbag
import argparse
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError


def bool_topic(topic):
    if(topic == "/cam_left_back/csi_cam/image_raw/compressed"):
        return 0
    elif (topic == "/cam_right_back/csi_cam/image_raw/compressed"):
        return 0
    elif (topic == "/cam_right_front/csi_cam/image_raw/compressed"):
        return 0
    elif (topic == "/cam_left_front/csi_cam/image_raw/compressed"):
        return 0
    elif (topic == "/cam_front_center/csi_cam/image_raw/compressed"):
        return 1
    elif (topic == "/cam_back/csi_cam/image_raw/compressed"):
        return 2
    elif (topic == "/cam_right_fish/csi_cam/image_raw/compressed"):
        return 0
    elif (topic == "/cam_front_fish/csi_cam/image_raw/compressed"):
        return 0
    elif (topic == "/cam_back_fish/csi_cam/image_raw/compressed"):
        return 0
    elif (topic == "/cam_left_fish/csi_cam/image_raw/compressed"):
        return 0
    else : 
        return 0
        
class ImageCreate():
    def __init__(self,filename):
        self.bridge = CvBridge()
        topic_LIST = [0,0,0,0,0,0,0,0,0,0,0]
        print(filename)
        with rosbag.Bag(filename,'r') as bag:
            for topic,msg,t in bag.read_messages():
                if bool_topic(topic) >= 1:
                    cv_path = path + topic.split('/')[1]
                    topic_LIST[bool_topic(topic)] += 1
                    if topic_LIST[bool_topic(topic)] % num_count != 0 :
                        continue
                    try:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg,"bgr8")
                    except CvBridgeError as e:
                        print (e)
                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    image_name = timestr+".png"
                    if not os.path.exists(cv_path):
                        os.makedirs(cv_path)
                    cv2.imwrite(cv_path+'/'+image_name,cv_image)
                
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='tool_rosbag_to_image.py')
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


    path=r'/mnt/data_1/Data/rosbag/images/qunguang_2021-12-07-11-47-22'
    path2 =  r'/mnt/data_1/Data/rosbag/qunguang/20211209/'
    num_count = 15
    for filename in os.listdir(path2):
        paron = r"1209_qunguang1_2_2021-12-09-11-01-17_1.bag"
        if re.search(paron,filename) != None:
            print(os.path.join(path2, filename))
            try:
                image_creator = ImageCreate(os.path.join(path2, filename))
            except rospy.ROSInterruptException:
                pass
