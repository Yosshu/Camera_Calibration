import numpy as np
import cv2
import glob
import os


# 3次元座標を描画

def draw(img, corners, imgpts):
    #corner = tuple(corners[0].ravel())
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255,0,0), 5)   # x
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 5)   # y
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 5)   # z
    return img


tate = 7
yoko = 10
ret2 = False    # 交点を見つけたかどうか

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((tate*yoko,3), np.float32)
objp[:,:2] = np.mgrid[0:yoko,0:tate].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(0)

#繰り返しのためのwhile文
while True:
    #カメラからの画像取得
    ret1, frame = cap.read()

    img = frame.copy()
    img_axes = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ret, img_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Find the chess board corners　交点を見つける
    ret2, corners = cv2.findChessboardCorners(gray, (yoko,tate),None)
    # If found, add object points, image points (after refining them)　交点が見つかったなら、描画
    if ret2 == True:
        cv2.imshow('frame',frame)
        objpoints.append(objp)      # object point

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) # 精度を上げている
        imgpoints.append(corners2)


        # パラメータの表示
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (yoko,tate), corners2,ret2)
        #cv2.imshow('drawChessboardCorners',img)
        cv2.waitKey(500)
        ret3, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        """
        ret：
        mtx：camera matrix，カメラ行列(内部パラメータ)
        dist：distortion coefficients，レンズ歪みパラメータ
        rvecs：rotation vectors，回転ベクトル
        tvecs：translation vectors，並進ベクトル
        """
        #print("ret: " + str(ret3) + "\nmtx: " + str(mtx) + "\ndist: " + str(dist) + "\nrvecs: " +  str(rvecs[-1]) + "\ntvecs: " + str(tvecs[-1]))
        #print()


        # 軸の定義
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        # project 3D points to image plane


        imgpts, jac = cv2.projectPoints(axis, rvecs[-1], tvecs[-1], mtx, dist)
        img_axes = draw(img_axes,corners2,imgpts)
        # img_axes = cv2.drawFrameAxes(img_axes, mtx, dist, rvecs[-1], tvecs[-1], 3, 3)
        cv2.imshow('Axes',img_axes)
        cv2.imwrite('axes.png',img_axes)



    #カメラの画像の出力
    cv2.imshow('camera' , frame)
    #繰り返し分から抜けるためのif文
    key =cv2.waitKey(1)
    if key == 27:   #Escで終了
        break

#メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()



"""
【参考】
カメラキャリブレーション
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_calib3d/py_calibration/py_calibration.html
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

姿勢推定
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_calib3d/py_pose/py_pose.html#pose-estimation
https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
"""