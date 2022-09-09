import cv2
import numpy as np
import sympy

cap = cv2.VideoCapture(0)          #カメラの設定　デバイスIDは0
while True:
    ret, frame = cap.read()           #カメラからの画像取得
    cv2.imshow('camera', frame)
    key =cv2.waitKey(0)
    print(key)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
