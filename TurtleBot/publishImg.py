#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16MultiArray

import numpy as np
import cv2

def talker(num):
    num = num.reshape(-1) #1次元に並べる
    pub_array = Int16MultiArray(data=num)
    pub = rospy.Publisher('img_nparray', Int16MultiArray, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    #r = rospy.Rate(10) # 10hz
    #rospy.loginfo(pub_array)
    pub.publish(pub_array)


#カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(0)
count = 0
#繰り返しのためのwhile文
while True:
    #カメラからの画像取得
    _, frame = cap.read()
    frame = cv2.resize(frame,dsize=(frame.shape[1]//4,frame.shape[0]//4)) #frame shape(270, 480, 3)
    #カメラの画像の出力
    cv2.imshow('camera', frame)
    #frame = np.array([[1, 2], [3, 4]]) #配列送信確認用

    #繰り返し分から抜けるためのif文
    key = cv2.waitKey(1)
    if key == ord('a'):                 # 写真撮影
        """
        img = frame.reshape(-1)
        f = open('imgarray.txt', 'w', encoding='UTF-8')
        for i in img:
            f.write(str(i)+", ")

        f.close()
        """
        try:
            talker(frame)
        except rospy.ROSInterruptException: pass
    elif key == 27:   #Escで終了
        break


#メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()