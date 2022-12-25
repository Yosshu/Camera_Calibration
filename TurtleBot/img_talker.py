#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Int16MultiArray
import numpy as np
import cv2

import time

import pyrealsense2 as rs

class testNode():
    def __init__(self):
        # Subscriberの作成
        self.sub = rospy.Subscriber('click_data', Int16MultiArray, self.callback)
        # Publisherの作成
        self.pub = rospy.Publisher('img_data', Int16MultiArray, queue_size=1)

    def callback(self, data):
        # callback関数の処理をかく
        #self.publish(data)
        lis_xy = np.array(data.data)
        print(lis_xy)

    def publish(self, data):
        data_array = data.reshape(-1)
        pub_array = Int16MultiArray(data=data_array)
        self.pub.publish(pub_array)


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    # Alignオブジェクト生成
    align_to = rs.stream.color
    align = rs.align(align_to)

    
    rospy.init_node('test_node')

    time.sleep(3.0)
    node = testNode()

    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print(depth_frame.get_distance(320,240))
        print(depth_image[240][320])

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #images = np.hstack((color_image, depth_colormap))
        #cv2.imshow("RealsenseImage",images)

        cv2.imshow("color_image",color_image)
        cv2.imshow("depth_colormap",depth_colormap)
        node.publish(color_image)
        key = cv2.waitKey(1)
        if key == 27:
            break
        rospy.sleep(0.1)

    cam.release()
    cv2.destroyAllWindows()

