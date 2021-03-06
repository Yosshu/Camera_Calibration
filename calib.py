import numpy as np
import cv2
import glob

tate = 7
yoko = 10

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
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ret, img_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Find the chess board corners　交点を見つける
    ret, corners = cv2.findChessboardCorners(gray, (yoko,tate),None)
    # If found, add object points, image points (after refining them)　交点が見つかったなら、描画
    if ret == True:
        cv2.imshow('frame',frame)
        objpoints.append(objp)      # object point

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)


        # パラメータの表示
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (yoko,tate), corners2,ret)
        cv2.imshow('drawChessboardCorners',img)
        cv2.waitKey(500)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        """
        ret：
        mtx：camera matrix，カメラ行列(内部パラメータ)
        dist：distortion coefficients，レンズ歪みパラメータ
        rvecs：rotation vectors，回転ベクトル
        tvecs：translation vectors，並進ベクトル
        """
        print("ret: " + str(ret) + "\nmtx: " + str(mtx) + "\ndist: " + str(dist) + "\nrvecs: " +  str(rvecs[-1]) + "\ntvecs: " + str(tvecs[-1]))



        # 歪み補正の準備
        img2 = frame.copy()
        h,  w = img2.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        print()


        # cv2.undistort() 関数を使う
        # undistort
        dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('frame.png',frame)
        cv2.imwrite('undistort.png',dst)
        #カメラの画像の出力
        cv2.imshow('undistort' , dst)



        # 再投影誤差(Re-projection Error)
        # パラメータを評価する
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(objpoints)) )    # 0に近ければ近いほど良い

        print("\n\n")



    #カメラの画像の出力
    cv2.imshow('camera' , frame)

    #繰り返し分から抜けるためのif文
    key =cv2.waitKey(10)
    if key == 27:   #Escで終了
        break

#メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()


"""
【参考】
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_calib3d/py_calibration/py_calibration.html
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""