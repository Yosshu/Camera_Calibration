#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray
import numpy as np
import cv2



def onMouse(event, x, y, flags, params):      # 1カメの画像に対するクリックイベント
    #global lis_array2
    if event == cv2.EVENT_LBUTTONDOWN:      # 左クリック
        z = params[y][x]
        if z == 0:
            print("人参しりしり")
        else:
            print(f'座標({x},{y})までの距離は{z/1000}[m]です！！！！！')
        

def callback(data):
    #global lis_array2
    lis_array = np.array(data.data)
    #print(lis_array[0:5])
    
    lis_array = lis_array.astype(np.uint16)
    #送る画像サイズで調節(例：lis_array = lis_array.reshape([1080,1920,3]))
    lis_array1 = lis_array[0:480*640*3]
    lis_array2 = lis_array[921600:]
    lis_array1 = lis_array1.astype(np.uint8)
    lis_array1 = lis_array1.reshape([480,640,3])
    lis_array2 = lis_array2.reshape([480,640])
    cv2.imshow('color_image',lis_array1)
    #cv2.imshow('depth_colormap',lis_array2)
    cv2.setMouseCallback('color_image',onMouse, lis_array2)
    cv2.waitKey(1)
    

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('img_data', Int32MultiArray, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
