#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import Int16
import numpy as np
import cv2


class depthimg():
    def __init__(self):
        # Subscriberの作成
        self.sub = rospy.Subscriber('img_data', Int16MultiArray, self.callback)
        # Publisherの作成
        self.pub = rospy.Publisher('depth_data', Int16, queue_size=1)

    def callback(self,data):
        tate = 240
        yoko = 320
        lis_array = np.array(data.data)

        lis_array = lis_array.astype(np.uint8)
        #送る画像サイズで調節(例：lis_array = lis_array.reshape([1080,1920,3]))
        lis_array1 = lis_array[0:tate*yoko*3]
        lis_array21 = lis_array[tate*yoko*3:tate*yoko*4]
        lis_array21 = lis_array21*256
        lis_array22 = lis_array[tate*yoko*4:]
        lis_array2 = lis_array21+lis_array22
        lis_array1 = lis_array1.reshape([tate,yoko,3])
        lis_array2 = lis_array2.reshape([tate,yoko])
        lis_array1 = cv2.resize(lis_array1,dsize=(640,480))
        lis_array2 = cv2.resize(lis_array2,dsize=(640,480))
        cv2.imshow('color_image',lis_array1)
        #lis_array2 = cv2.applyColorMap(cv2.convertScaleAbs(lis_array2, alpha=0.03),cv2.COLORMAP_JET)
        #cv2.imshow('depth_colormap',lis_array2)
        cv2.setMouseCallback('color_image',self.onMouse_lis, lis_array2)
        cv2.waitKey(1)

    def publish(self, data):
        self.pub.publish(data)

    def onMouse_lis(self, event, x, y, flags, params):      # 1カメの画像に対するクリックイベント
        if event == cv2.EVENT_LBUTTONDOWN:      # 左クリック
            z = params[y][x]
            if z == 0:
                print("Error")
            else:
                print(f'座標({x},{y})までの距離は{z/1000}[m]です')
                self.publish(z//10)



if __name__ == '__main__':
    rospy.init_node('listener')

    #time.sleep(3.0)
    dep = depthimg()

    while not rospy.is_shutdown():
        rospy.sleep(0.1)

