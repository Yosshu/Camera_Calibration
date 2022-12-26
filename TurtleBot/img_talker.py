#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int16MultiArray
import numpy as np
import cv2

import pyrealsense2 as rs

def talker(data_array1,data_array2):
    data_array1 = data_array1.reshape(-1)
    data_array2 = data_array2.reshape(-1)
    data_array21 = data_array2//256
    data_array22 = data_array2%256
    data_array3 = np.hstack((data_array1, data_array21, data_array22))
    data_array3 = data_array3.astype(int)
    pub_array = Int16MultiArray(data=data_array3)
    pub.publish(pub_array)

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)    
    pub = rospy.Publisher('img_data', Int16MultiArray, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(5)
    key = -1
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Alignオブジェクト生成
    align_to = rs.stream.color
    align = rs.align(align_to)

    # decimarion_filterのパラメータ
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 1)
    # spatial_filterのパラメータ
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    # hole_filling_filterのパラメータ
    hole_filling = rs.hole_filling_filter()
    # disparity
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)


    while not rospy.is_shutdown():
        """
        _, img = cam.read()
        img2 = cv2.resize(img, (480,270))
        cv2.imshow('pub_img',img2)
        key = cv2.waitKey(1)
        """
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        # filterをかける
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue


        # filterをかける
        filter_frame = decimate.process(depth_frame)
        filter_frame = depth_to_disparity.process(filter_frame)
        filter_frame = spatial.process(filter_frame)
        filter_frame = disparity_to_depth.process(filter_frame)
        filter_frame = hole_filling.process(filter_frame)
        depth_frame = filter_frame.as_depth_frame()

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #images = np.hstack((color_image, depth_colormap))
        #cv2.imshow("RealsenseImage",images)
        cv2.imshow("color_image",color_image)
        cv2.imshow("depth_colormap",depth_colormap)
        #cv2.imshow("depth_colormap",depth_colormap)
        key = cv2.waitKey(1)
        if key == 27:
            break
        data_array1 = np.array(color_image)
        data_array2 = np.array(depth_image)
        #print(color_image.shape)
        talker(data_array1, data_array2)
        rate.sleep()

    
    cam.release()
    cv2.destroyAllWindows()
    