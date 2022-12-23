#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int16MultiArray
import numpy as np
import cv2

def talker(data_array):
    data_array = data_array.reshape(-1)
    pub_array = Int16MultiArray(data=data_array)
    pub.publish(pub_array)
    rate.sleep()

cam = cv2.VideoCapture(0)    
pub = rospy.Publisher('img_data', Int16MultiArray, queue_size=10)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(5)
key = -1

if __name__ == '__main__':
    while not rospy.is_shutdown():
        _, img = cam.read()
        img2 = cv2.resize(img, (480,270))
        cv2.imshow('pub_img',img2)
        key = cv2.waitKey(1)
        if key == 27:
            break
        data_array = np.array(img2)
        talker(data_array)
    
    cam.release()
    cv2.destroyAllWindows()
    