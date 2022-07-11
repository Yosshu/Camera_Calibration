from re import X
import numpy as np
import cv2
import glob
import pyautogui

# 検出するチェッカーボードの交点の数
tate = 7
yoko = 10


# 3次元座標を描画
def draw(img, corners, imgpts):
    #corner = tuple(corners[0].ravel())
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255,0,0), 5)   # x Blue
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 5)   # y Green
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 5)   # z Red
    return img

# 画像のどこをクリックしたか返す
def onMouse(event, x, y, flags, params):
    global mtx, mtx2, tvecs, tvecs2, R, R2, img_axes2
    if event == cv2.EVENT_LBUTTONDOWN:
        obj_i1x = x                                                         # 対象物の1カメ画像座標　クリックした点
        obj_i1y = y
        obj_n1x = (obj_i1x - mtx[0][2]) / mtx[0][0]                         # 対象物の1カメ正規化座標　原点を真ん中にしてから，焦点距離で割る
        obj_n1y = (obj_i1y - mtx[1][2]) / mtx[1][1]
        obj_n1 = [[obj_n1x], [obj_n1y], [1]]                                # 対象物の1カメ正規化画像座標系を1カメカメラ座標系に変換　(x_normimg, y_normimg, 1)
        obj_w = (np.linalg.inv(R)) @ (np.array(obj_n1) - np.array(tvecs))      # ncoordを世界座標系に変換              Ｗ = Ｒ^T (Ｃ1 - ｔ)

        obj_c2 = np.array(R2) @ np.array(obj_w) + np.array(tvecs2)          # wcoordを2カメのカメラ座標系に変換     Ｃ2 = ＲＷ + ｔ
        obj_n2x = obj_c2[0][0][0]/obj_c2[0][2][0]                           # ccoord2を2カメの正規化画像座標系に変換　(Xc/Zc, Yc/Zc, 1)
        obj_n2y = obj_c2[0][1][0]/obj_c2[0][2][0]
        obj_i2x = obj_n2x * mtx2[0][0] + mtx2[0][2]                         # 正規化画像座標を2カメの画像座標系に変換　焦点距離を掛けてから，原点を左上にする
        obj_i2y = obj_n2y * mtx2[1][1] + mtx2[1][2]
        #print(rvecs)
        #print(R)


        camera1_w = (np.linalg.inv(R)) @ (np.array([[0], [0], [0]]) - np.array(tvecs))     # 1カメのワールド座標        Ｗ = Ｒ^T (Ｃ1 - ｔ)
        camera1_c2 = np.array(R2) @ camera1_w + np.array(tvecs)                         # 2カメを原点としたカメラ座標系での1カメの位置
        camera1_n2 = camera1_c2 / camera1_c2[0][2][0]                                   # 2カメの正規化画像座標系での1カメの位置
        camera1_i2x = camera1_n2[0][0][0] * mtx2[0][0] + mtx2[0][2]                     # 2カメの画像座標系での1カメの位置
        camera1_i2y = camera1_n2[0][1][0] * mtx2[1][1] + mtx2[1][2]

        img_line = img_axes2.copy()

        slope = (camera1_i2y - obj_i2y)/(camera1_i2x - obj_i2x)

        #if camera1_c2 - obj_i2y > camera1_i2x - obj_i2x:


        a = 0
        dx = -1 * camera1_i2x - a
        dy = slope * dx

        #koko
        dx2 = -1 * camera1_i2x + 640 + a
        dy2 = slope * dx2
        #kookmade

        img_line = cv2.line(img_line, (int(camera1_i2x + dx), int(camera1_i2y + dy)), (int(camera1_i2x + dx2), int(camera1_i2y + dy2)), (0,255,255), 5)
        #img_line = cv2.line(img_line, (int(camera1_i2x + dx), int(camera1_i2y + dy)), (int(obj_i2x - dx), int(obj_i2y - dy)), (0,255,255), 5)
        cv2.imshow('Axes2',img_line)
        #print(int(obj_i2x), int(obj_i2y))

        print((int(camera1_i2x + dx), int(camera1_i2y + dy)), (int(camera1_i2x + dx2), int(camera1_i2y + dy2)))
        





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





    # 回転行列を3×1から3×3に変換
    R, Jacob = cv2.Rodrigues(np.array(rvecs))
    R2, Jacob2 = cv2.Rodrigues(np.array(rvecs2))
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
    
    
    cv2.setMouseCallback('Axes', onMouse)



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

画像座標系からカメラ座標系への変換
https://mem-archive.com/2018/10/13/post-682/
"""