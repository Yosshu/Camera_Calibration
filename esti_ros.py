from re import X
import numpy as np
import cv2
import glob
import pyautogui
import copy
import math


def draw(img, corners, imgpts):         # 座標軸を描画する関数
    #corner = tuple(corners[0].ravel())
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255,0,0), 3)   # X軸 Blue
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 3)   # Y軸 Green
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 3)   # Z軸 Red
    return img

def axes_check(img, tate, yoko, objp, criteria, axis):      # Z軸が正しく伸びているかを確認するための関数
    objpoints0 = []
    imgpoints0 = []
    gray0 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret0, corners0 = cv2.findChessboardCorners(gray0, (yoko,tate),None)
    if ret0 == True:
        objpoints0.append(objp)      # object point
        corners02 = cv2.cornerSubPix(gray0,corners0,(11,11),(-1,-1),criteria) # 精度を上げている
        imgpoints0.append(corners02)
        cv2.waitKey(500)
        ret, mtx0, dist0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints0, imgpoints0, gray0.shape[::-1],None,None)
        # Find the rotation and translation vectors.
        _, rvecs0, tvecs0, _ = cv2.solvePnPRansac(objp, corners02, mtx0, dist0)

        imgpts0, _ = cv2.projectPoints(axis, rvecs0, tvecs0, mtx0, dist0)
        return ret0, corners02, imgpts0
    return ret0, None, None

def drawpoints(img, points,b,g,r):
    for i in points:
        if 0<=i[0]<img.shape[1] and 0<=i[1]<img.shape[0]:
            img = cv2.circle(img, (int(i[0]),int(i[1])), 2, (b, g, r), thickness=-1)
    return img

class Estimation:
    def __init__(self, mtx, dist, rvecs, tvecs, img, imgpoints, tate, yoko):
        self.mtx = mtx                  # 1カメの内部パラメータ
        self.dist = dist                # 1カメの歪み係数
        self.rvecs = rvecs              # 1カメの回転ベクトル
        self.tvecs = tvecs              # 1カメの並進ベクトル

        """
        self.k1_1 = dist[0][0] 
        self.k2_1 = dist[0][1] 
        self.p1_1 = dist[0][2] 
        self.p2_1 = dist[0][3] 
        self.k3_1 = dist[0][4] 
        self.k4_1 = dist[0][5] 
        self.k5_1 = dist[0][6] 
        self.k6_1 = dist[0][7] 

        self.k1_2 = dist2[0][0] 
        self.k2_2 = dist2[0][1] 
        self.p1_2 = dist2[0][2] 
        self.p2_2 = dist2[0][3] 
        self.k3_2 = dist2[0][4] 
        self.k4_2 = dist2[0][5] 
        self.k5_2 = dist2[0][6] 
        self.k6_2 = dist2[0][7] 
        """

        """
        h,  w = img.shape[:2]
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        self.newcameramtx2, self.roi2 = cv2.getOptimalNewCameraMatrix(mtx2,dist2,(w,h),1,(w,h))
        """


        self.imgpoints = imgpoints      # 1カメで見つかったチェッカーボードの交点の画像座標
        self.tate = tate                # 検出するチェッカーボードの交点の縦の数
        self.yoko = yoko                # 検出するチェッカーボードの交点の横の数
        
        # 回転ベクトルを3×1から3×3に変換
        self.R, _ = cv2.Rodrigues(np.array(self.rvecs))     # 1カメの回転行列

        self.click_count = 0            # 1カメの画像をクリックしたか

        # クラス内の関数間で共有したい変数
        self.obj1_i1x = 0               # 1カメでクリックした点の1カメ画像座標
        self.obj1_i1y = 0
        self.img = img      # 軸だけ描画された1カメの画像
        self.pn = [1, 1, 1]             # 1 or -1，ワールド座標がプラスかマイナスか，出力する直前にかける [X軸座標, Y軸座標, Z軸座標]
        self.origin = []

        self.camera1_w = []             # 1カメのワールド座標
        self.obj1_w = []                # 1カメでクリックした点のワールド座標


    def onMouse(self, event, x, y, flags, params):      # 1カメの画像に対するクリックイベント
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pointFixZ(x,y,1)

    def line_SEpoint(self, x, y, num):      # 始点（カメラ）と終点（正規化画像座標）のワールド座標を求める関数，numは1カメか2カメか
        obj_i = [x,y]
        if num == 1:
            obj_sphi1 = self.undist_pts(np.array([obj_i],dtype='float32'),1)

            obj_n1x = (obj_sphi1[0] - self.mtx[0][2]) / self.mtx[0][0]                   # 対象物の1カメ正規化座標　原点を真ん中にしてから，焦点距離で割る
            obj_n1y = (obj_sphi1[1] - self.mtx[1][2]) / self.mtx[1][1]
            
            obj_n1 = [[obj_n1x], [obj_n1y], [1]]                                    # 対象物の1カメ正規化画像座標系を1カメカメラ座標系に変換
            obj1_w = (np.linalg.inv(self.R)) @ (np.array(obj_n1) - np.array(self.tvecs))
            #obj1_w = (self.R.T) @ (np.array(obj_n1) - np.array(self.tvecs))                        # obj_n1を世界座標系に変換              Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
            
            #camera1_w = (np.linalg.inv(R)) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))     # 1カメのワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
            camera1_w = (self.R.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))             # 1カメのワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)

            return camera1_w, obj1_w

        elif num == 2:
            obj_sphi2 = self.undist_pts(np.array([obj_i],dtype='float32'),2)

            obj_n2x = (obj_sphi2[0] - self.mtx2[0][2]) / self.mtx2[0][0]                 # 対象物の2カメ正規化座標　原点を真ん中にしてから，焦点距離で割る
            obj_n2y = (obj_sphi2[1] - self.mtx2[1][2]) / self.mtx2[1][1]
            
            obj_n2 = [[obj_n2x], [obj_n2y], [1]]                                    # 対象物の2カメ正規化画像座標系を2カメカメラ座標系に変換
            #obj1_w = (np.linalg.inv(R)) @ (np.array(obj_n1) - np.array(tvecs))
            obj2_w = (self.R2.T) @ (np.array(obj_n2) - np.array(self.tvecs2))                       # obj_n2を世界座標系に変換              Ｗ = Ｒ2^T (Ｃ2 - ｔ2)
            
            #camera2_w = (np.linalg.inv(R2)) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs2))        # 2カメのワールド座標        Ｗ = Ｒ2^T (Ｃ2 - ｔ2)
            camera2_w = (self.R2.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs2))           # 2カメのワールド座標        Ｗ = Ｒ2^T (Ｃ2 - ｔ2)

            return camera2_w, obj2_w

        return None, None


    def undist_npoint(self, x, y, num):
        r = math.sqrt(x**2 + y**2)
        if num == 1:
            nume = 1 + self.k1_1*r**2 + self.k2_1*r**4 + self.k3_1*r**6
            deno = 1 + self.k4_1*r**2 + self.k5_1*r**4 + self.k6_1*r**6
            undist_n1x = x*(nume/deno) + 2*self.p1_1*x*y + self.p2_1*(r**2 + 2*x**2)
            undist_n1y = y*(nume/deno) + self.p1_1*(r**2 + 2*y**2) + 2*self.p2_1*x*y
            return undist_n1x, undist_n1y
        elif num == 2:
            nume = 1 + self.k1_2*r**2 + self.k2_2*r**4 + self.k3_2*r**6
            deno = 1 + self.k4_2*r**2 + self.k5_2*r**4 + self.k6_2*r**6
            undist_n2x = x*(nume/deno) + 2*self.p1_2*x*y + self.p2_2*(r**2 + 2*x**2)
            undist_n2y = y*(nume/deno) + self.p1_2*(r**2 + 2*y**2) + 2*self.p2_2*x*y
            return undist_n2x, undist_n2y
        return None, None


    def line_update(self, img, num):        # エピポーラ線やクリックした点を描画する関数
        if num == 2:                                                                                                    # 2カメ画像に対しての処理の場合
            if self.click_count >= 1:                                                                                   # 1カメ画像をクリックしていたら，
                startpoint_i2y, endpoint_i2y = self.epipo(self.camera1_w, self.obj1_w, 1)                               # エピポーラ線を求める
                img = cv2.line(img, (0, int(startpoint_i2y)), (img.shape[1], int(endpoint_i2y)), (0,255,255), 2)        # エピポーラ線を描画
                if self.click_count == 2:                                                                               # 2カメ画像をクリックしていたら，
                    img = cv2.circle(img, (int(self.obj2_i2x),int(self.obj2_i2y)), 4, (0, 165, 255), thickness=-1)      # クリックした点を描画
            #img = cv2.undistort(img, self.mtx2, self.dist2, None, self.newcameramtx2)
            return img

        elif num == 1:                                                                                                  # 1カメ画像に対しての処理の場合
            if self.click_count >= 1:                                                                                   # 1カメ画像をクリックしていたら，
                img = cv2.circle(img, (int(self.obj1_i1x),int(self.obj1_i1y)), 4, (0, 165, 255), thickness=-1)          # クリックした点を描画
                if self.click_count == 2:                                                                               # 2カメ画像をクリックしていたら，
                    startpoint_i1y, endpoint_i1y = self.epipo(self.camera2_w, self.obj2_w, 2)                           # エピポーラ線を求める
                    img = cv2.line(img, (0, int(startpoint_i1y)), (img.shape[1], int(endpoint_i1y)), (0,255,255), 2)    # エピポーラ線を描画
            #img = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
            return img

        return None


    def undist_pts(self, pts_uv, num):
        if num == 1:
            pts_uv = cv2.undistortPoints(pts_uv, self.mtx, self.dist, P=self.mtx)
            pts_uv = pts_uv[0][0]
            pts_uv = [pts_uv[0],pts_uv[1]]
        return pts_uv

    def pointFixZ(self,ix,iy,num):
        floor_wz = 0
        obj_i = [ix,iy]
        floor_wz = -floor_wz
        if num == 1:
            obj_sphi1 = self.undist_pts(np.array([obj_i],dtype='float32'),1)

            obj_n1x = (obj_sphi1[0] - self.mtx[0][2]) / self.mtx[0][0]                   # 対象物の1カメ正規化座標　原点を真ん中にしてから，焦点距離で割る
            obj_n1y = (obj_sphi1[1] - self.mtx[1][2]) / self.mtx[1][1]
            
            obj_n1 = [[obj_n1x], [obj_n1y], [1]]                                    # 対象物の1カメ正規化画像座標系を1カメカメラ座標系に変換
            obj1_w = (np.linalg.inv(self.R)) @ (np.array(obj_n1) - np.array(self.tvecs))
            #obj1_w = (self.R.T) @ (np.array(obj_n1) - np.array(self.tvecs))                        # obj_n1を世界座標系に変換              Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
            
            camera1_w = (np.linalg.inv(self.R)) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))     # 1カメのワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
            #camera1_w = (self.R.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))             # 1カメのワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
            
            slopexz_w = (camera1_w[0][2][0] - obj1_w[0][2][0])/(camera1_w[0][0][0] - obj1_w[0][0][0])
            floor_wx = ((floor_wz - camera1_w[0][2][0])/slopexz_w) + camera1_w[0][0][0]
            slopeyz_w = (camera1_w[0][2][0] - obj1_w[0][2][0])/(camera1_w[0][1][0] - obj1_w[0][1][0])
            floor_wy = ((floor_wz - camera1_w[0][2][0])/slopeyz_w) + camera1_w[0][1][0]
            floor_wx = round(floor_wx, 4)
            floor_wy = round(floor_wy, 4)
            print([floor_wx,floor_wy,-floor_wz])

def main():
    # 検出するチェッカーボードの交点の数
    tate = 4
    yoko = 5
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((tate*yoko,3), np.float32)
    objp[:,:2] = np.mgrid[0:yoko,0:tate].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # 軸の定義
    axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3)

    cap1 = cv2.VideoCapture(0)          #カメラの設定　デバイスIDは0

    val = None
    while True:
        val = input('Checkerboard or Prepared value? (C/P)')        # チェッカーボードを使うか用意した値を使うかを指定する
        if val == "c" or val == "C" or val == "p" or val == "P":    # CかPでなければ何度も聞く
            break

    if val == "c" or val == "C":            # チェッカーボードを使う場合
        img_axes0 = []
        imgpts0 = []
        corners02 = []

        ret01 = False
        
        while True:
            _, frame1 = cap1.read()           #カメラからの画像取得

            if ret01 == True:
                img_axes0 = frame1.copy()
                img_axes0 = draw(img_axes0,corners02,imgpts0)
                cv2.imshow('camera1' , img_axes0)
            else:
                cv2.imshow('camera1' , frame1)
            
            #繰り返し分から抜けるためのif文
            key =cv2.waitKey(1)
            if key == ord('s'):
                gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                _, img_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                # Find the chess board corners　交点を見つける
                ret, corners = cv2.findChessboardCorners(img_otsu, (yoko,tate),None)
                if ret == True:
                    print("OK")
                    break
                elif ret == False:
                    print("Fail")
            elif key == ord('a'):
                ret01, corners02, imgpts0 = axes_check(frame1, tate, yoko, objp, criteria, axis)
            elif key == 27:   #Escで終了
                break
        
        #cv2.imwrite('1.png',frame1)

        """
        frame1 = cv2.imread("1.png")
        cv2.imshow('camera1',frame1)
        """  
        
        gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        ret, img_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # Find the chess board corners　交点を見つける
        #cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(img_otsu, (yoko,tate), None)
        # If found, add object points, image points (after refining them)　交点が見つかったなら、描画
        corners12 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners12)

    elif val == "p" or val == "P":          # 用意した値を使う場合
        _, frame1 = cap1.read()           #カメラからの画像取得
        gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        cv2.imshow('camera1' , frame1)
        corners = np.array([290, 212, 338, 235, 395, 262, 457, 292, 236, 242, 284, 268, 340, 300, 404, 336, 180, 276, 225, 306, 279, 343, 341, 386],dtype='float32').reshape(-1,1,2)
        corners12 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners12)

    objpoints.append(objp)      # object point

    # パラメータの表示
    # Draw and display the corners
    #img = cv2.drawChessboardCorners(img, (yoko,tate), corners12,ret)
    #img2 = cv2.drawChessboardCorners(img2, (yoko,tate), corners22,ret2)
    #cv2.imshow('drawChessboardCorners',img)
    cv2.waitKey(500)
    """
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None,flags=cv2.CALIB_RATIONAL_MODEL) # _, _ は，rvecs, tvecs
    """
    mtx = np.array([590, 0, frame1.shape[1]/2, 0, 590, frame1.shape[0]/2, 0, 0, 1]).reshape(3,3)

    dist = np.array([-0.55376925, 1.99642305, 0.00695332, -0.02167939, 2.74771938, 0.54082278, 0.04485288, 4.88112929, 0, 0, 0, 0, 0, 0]).reshape(1,-1)
    _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners12, mtx, dist)
    rvecs = [rvecs]
    tvecs = [tvecs]

    """
    ret：
    mtx：camera matrix，カメラ行列(内部パラメータ)
    dist：distortion coefficients，レンズ歪みパラメータ
    rvecs：rotation vectors，回転ベクトル
    tvecs：translation vectors，並進ベクトル
    """
    #print("ret: " + str(ret) + "\nmtx: " + str(mtx) + "\ndist: " + str(dist) + "\nrvecs: " +  str(rvecs[-1]) + "\ntvecs: " + str(tvecs[-1]))
    
    # project 3D points to image plane
    imgpts, _ = cv2.projectPoints(axis, rvecs[-1], tvecs[-1], mtx, dist)
    
    es = Estimation(mtx, dist, rvecs, tvecs, frame1, imgpoints, tate, yoko)


    cv2.setMouseCallback('camera1', es.onMouse)         # 1カメの画像に対するクリックイベント
    
    while True:
        ret, frame1 = cap1.read()           #カメラからの画像取得
        img_axes = draw(frame1,corners12,imgpts)
        img_axes = es.line_update(img_axes,1)
        cv2.imshow('camera1', img_axes)      #カメラの画像の出力
        
        #繰り返し分から抜けるためのif文
        key =cv2.waitKey(1)
        if key == 27:   #Escで終了
            cap1.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()

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
直線同士の最接近点
https://phst.hateblo.jp/entry/2020/02/29/000000
"""