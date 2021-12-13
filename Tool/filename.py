import os
import re
import cv_bridge
import roslib
import rosbag
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
#conda deactivate
path='image/'

class ImageCreate():
    def __init__(self,filename):
        self.bridge = CvBridge()
        i = 0
        # num = 0
        print(filename)
        with rosbag.Bag(filename,'r') as bag:
            for topic,msg,t in bag.read_messages():
                # print(i)
                if topic == "/cam_front_center/csi_cam/image_raw/compressed":
                    i += 1
                    if i%15 != 0 :
                        continue
                    try:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg,"bgr8")
                    except CvBridgeError as e:
                        print (e)
                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    image_name = timestr+".png"
                    # if not os.exist(path):
	                #     os.makedirs(path)
                    cv2.imwrite(path+image_name,cv_image)
                    # time.sleep(0.9)
            # print("=================",num)


if __name__ == '__main__':
    
    path2 = r'/media/ljp/vision/ljp/1207'
    for filename in os.listdir(path2):
        paron = r".bag"
        if re.search(paron,filename) != None:
            print(os.path.join(path2, filename))
            try:
                image_creator = ImageCreate(os.path.join(path2, filename))
            except rospy.ROSInterruptException:
                pass
