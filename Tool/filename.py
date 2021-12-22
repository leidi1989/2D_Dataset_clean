'''
Description: 
Version: 
Author: Leidi
Date: 2021-12-08 16:34:03
LastEditors: Leidi
LastEditTime: 2021-12-22 11:08:12
'''
import os
import re
import rosbag
import rospy
import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError


path='image/'

class ImageCreate():
    def __init__(self,filename):
        self.bridge = CvBridge()
        i = 0
        print(filename)
        with rosbag.Bag(filename,'r') as bag:
            for topic,msg,t in bag.read_messages():
                if topic == '/cam_front_center/csi_cam/image_raw/compressed':
                    i += 1
                    if i%15 != 0 :
                        continue
                    try:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg,'bgr8')
                    except CvBridgeError as e:
                        print (e)
                    timestr = '%.6f' % msg.header.stamp.to_sec()
                    image_name = timestr+'.png'
                    cv2.imwrite(path+image_name,cv_image)


if __name__ == "__main__":
    
    path2 = r'/media/ljp/vision/ljp/1207'
    for filename in os.listdir(path2):
        paron = r'.bag'
        if re.search(paron,filename) != None:
            print(os.path.join(path2, filename))
            try:
                image_creator = ImageCreate(os.path.join(path2, filename))
            except rospy.ROSInterruptException:
                pass
