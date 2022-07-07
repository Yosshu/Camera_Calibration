import numpy as np
import cv2
import glob
import pyautogui

tate = 7
yoko = 10

# 3次元座標を描画
def draw(img, corners, imgpts):
    #corner = tuple(corners[0].ravel())
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255,0,0), 5)   # x Blue
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 5)   # y Green
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 5)   # z Red
    return img


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((tate*yoko,3), np.float32)
objp[:,:2] = np.mgrid[0:yoko,0:tate].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
imgpoints2 = [] # 2d points in image plane.

frame = cv2.imread('1.png')  #queryimage # left image
frame2 = cv2.imread('2.png')  #queryimage # left image

img = frame.copy()
img_axes = frame.copy()

img2 = frame2.copy()
img_axes2 = frame2.copy()


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, img_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret2, img_otsu2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)

# Find the chess board corners　交点を見つける
ret, corners = cv2.findChessboardCorners(img_otsu, (yoko,tate),None)
ret2, corners2 = cv2.findChessboardCorners(img_otsu2, (yoko,tate),None)
# If found, add object points, image points (after refining them)　交点が見つかったなら、描画
if ret and ret2 == True:
    objpoints.append(objp)      # object point
    corners12 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    corners22 = cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
    imgpoints.append(corners12)
    imgpoints2.append(corners22)
    # パラメータの表示
    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (yoko,tate), corners12,ret)
    img2 = cv2.drawChessboardCorners(img2, (yoko,tate), corners22,ret2)
    #cv2.imshow('drawChessboardCorners',img)
    cv2.waitKey(500)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1],None,None)
    """
    ret：
    mtx：camera matrix，カメラ行列(内部パラメータ)
    dist：distortion coefficients，レンズ歪みパラメータ
    rvecs：rotation vectors，回転ベクトル
    tvecs：translation vectors，並進ベクトル
    """
    #print("ret: " + str(ret) + "\nmtx: " + str(mtx) + "\ndist: " + str(dist) + "\nrvecs: " +  str(rvecs[-1]) + "\ntvecs: " + str(tvecs[-1]))



    # 軸の定義
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    # project 3D points to image plane


    imgpts, jac = cv2.projectPoints(axis, rvecs[-1], tvecs[-1], mtx, dist)
    img_axes = draw(img_axes,corners12,imgpts)
    # img_axes = cv2.drawFrameAxes(img_axes, mtx, dist, rvecs[-1], tvecs[-1], 3, 3)
    cv2.imshow('Axes',img_axes)

    imgpts2, jac2 = cv2.projectPoints(axis, rvecs2[-1], tvecs2[-1], mtx2, dist2)
    img_axes2 = draw(img_axes2,corners22,imgpts2)
    # img_axes = cv2.drawFrameAxes(img_axes, mtx, dist, rvecs[-1], tvecs[-1], 3, 3)
    cv2.imshow('Axes2',img_axes2)


#繰り返し分から抜けるためのif文
key =cv2.waitKey(0)
if key == 27:   #Escで終了
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