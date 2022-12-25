#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Int16MultiArray
import numpy as np
import cv2

import time

class testNode():
    def __init__(self):
        # Subscriberの作成
        self.sub = rospy.Subscriber('img_data', Int16MultiArray, self.callback)
        # Publisherの作成
        self.pub = rospy.Publisher('click_data', Int16MultiArray, queue_size=1)

    def callback(self, data):
        # callback関数の処理をかく
        lis_array = np.array(data.data)
        lis_array = lis_array.astype(np.uint8)
        #送る画像サイズで調節(例：lis_array = lis_array.reshape([1080,1920,3]))
        lis_array = lis_array.reshape([480,640,3])
        cv2.imshow('lis_img',lis_array)
        cv2.setMouseCallback('lis_img',self.onMouse)
        cv2.waitKey(1)


    def publish(self, x,y):
        data_array = [x,y]
        pub_array = Int16MultiArray(data=data_array)
        self.pub.publish(pub_array)

    def onMouse(self, event, x, y, flags, params):      # 1カメの画像に対するクリックイベント
        if event == cv2.EVENT_LBUTTONDOWN:      # 左クリック
            self.publish(x,y)

if __name__ == '__main__':
    rospy.init_node('test_node2')

    time.sleep(3.0)
    node = testNode()

    while not rospy.is_shutdown():
        rospy.sleep(0.1)