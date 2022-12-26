#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int16MultiArray
import numpy as np
import cv2



def onMouse(event, x, y, flags, params):      # 1カメの画像に対するクリックイベント
    #global lis_array2
    if event == cv2.EVENT_LBUTTONDOWN:      # 左クリック
        z = params[y][x]
        if z == 0:
            print("orz")
        else:
            print(f'座標({x},{y})までの距離は{z/1000}[m]です!!!!!!!')


def callback(data):
    lis_array = np.array(data.data)

    lis_array = lis_array.astype(np.uint8)
    #送る画像サイズで調節(例：lis_array = lis_array.reshape([1080,1920,3]))
    lis_array1 = lis_array[0:921600]
    lis_array21 = lis_array[921600:1228800]
    lis_array21 = lis_array21*256
    lis_array22 = lis_array[1228800:]
    lis_array2 = lis_array21+lis_array22
    lis_array1 = lis_array1.reshape([480,640,3])
    lis_array2 = lis_array2.reshape([480,640])
    cv2.imshow('color_image',lis_array1)
    #lis_array2 = cv2.applyColorMap(cv2.convertScaleAbs(lis_array2, alpha=0.03),cv2.COLORMAP_JET)
    #cv2.imshow('depth_colormap',lis_array2)
    cv2.setMouseCallback('color_image',onMouse, lis_array2)
    cv2.waitKey(1)
    

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('img_data', Int16MultiArray, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
