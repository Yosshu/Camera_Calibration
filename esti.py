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
        imgpts0, _ = cv2.projectPoints(axis, rvecs0[-1], tvecs0[-1], mtx0, dist0)
        return ret0, corners02, imgpts0
    return ret0, None, None

class Estimation:
    def __init__(self, mtx, dist, rvecs, tvecs, mtx2, dist2, rvecs2, tvecs2, img_axes2, imgpoints, imgpoints2, tate, yoko):
        self.mtx = mtx                  # 1カメの内部パラメータ
        self.dist = dist                # 1カメの歪み係数
        self.rvecs = rvecs              # 1カメの回転ベクトル
        self.tvecs = tvecs              # 1カメの並進ベクトル
        self.mtx2 = mtx2                # 2カメの内部パラメータ
        self.dist2 = dist2              # 2カメの歪み係数
        self.rvecs2 = rvecs2            # 2カメの回転ベクトル
        self.tvecs2 = tvecs2            # 2カメの並進ベクトル

        
        h,  w = img_axes2.shape[:2]
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        self.newcameramtx2, self.roi2 = cv2.getOptimalNewCameraMatrix(mtx2,dist2,(w,h),1,(w,h))


        self.imgpoints = imgpoints      # 1カメで見つかったチェッカーボードの交点の画像座標
        self.imgpoints2 = imgpoints2    # 2カメで見つかったチェッカーボードの交点の画像座標
        self.tate = tate                # 検出するチェッカーボードの交点の縦の数
        self.yoko = yoko                # 検出するチェッカーボードの交点の横の数
        
        # 回転ベクトルを3×1から3×3に変換
        self.R, _ = cv2.Rodrigues(np.array(self.rvecs))     # 1カメの回転行列
        self.R2, _ = cv2.Rodrigues(np.array(self.rvecs2))   # 2カメの回転行列

        self.click_count = 0            # 1カメの画像をクリックしたか

        # クラス内の関数間で共有したい変数
        self.obj1_i1x = 0               # 1カメでクリックした点の1カメ画像座標
        self.obj1_i1y = 0
        self.obj2_i2x = 0               # 2カメでクリックした点の2カメ画像座標
        self.obj2_i2y = 0
        self.img_axes2 = img_axes2      # 軸だけ描画された1カメの画像
        self.img_line = []              # 黄色の線を引いた2カメの画像
        self.SF = []                    # スケールファクタ [X軸座標, Y軸座標, Z軸座標]
        self.pn = [1, 1, 1]             # 1 or -1，ワールド座標がプラスかマイナスか，出力する直前にかける [X軸座標, Y軸座標, Z軸座標]
        self.origin = []

        self.camera1_w = []             # 1カメのワールド座標
        self.obj1_w = []                # 1カメでクリックした点のワールド座標
        self.camera2_w = []             # 2カメのワールド座標
        self.obj2_w = []                # 2カメでクリックした点のワールド座標

        self.ScaleFactor()              # スケールファクタを求める


    def ScaleFactor(self):       # スケールファクタを求める関数，結果をスケールファクタで割ることで1マスが1になるようにする
        stdnum_z = 2         # Z軸方向スケールファクタを求める時の基準点の個数は，tate*yoko*stdnum 個
        #print(self.imgpoints[0][60][0][1])
        #cv2.circle(self.img_axes2, (int(stdpoints2z[0][0][0]),int(stdpoints2z[0][0][1])), 8, (0, 165, 255), thickness=-1)
        #cv2.imshow("self.img_axes2", self.img_axes2)

        std_w = []          # チェッカーボードの交点のワールド座標を格納する配列
        for i in range(self.tate):
            for j in range(self.yoko):
                k = i*self.yoko + j
                camera1_w, obj1_w = self.line_SEpoint(self.imgpoints[0][k][0][0], self.imgpoints[0][k][0][1], 1)
                camera2_w, obj2_w = self.line_SEpoint(self.imgpoints2[0][k][0][0], self.imgpoints2[0][k][0][1], 2)
                line1x = np.hstack((camera1_w[0].T, obj1_w[0].T)).reshape(2, 3)
                line2x = np.hstack((camera2_w[0].T, obj2_w[0].T)).reshape(2, 3)
                res = self.distance_2lines(line1x, line2x)
                std_w.append(res)
        self.origin = std_w[0]  # 原点のワールド座標

        # X軸方向
        std_diffx = []
        for i in range(self.tate):
            for j in range(self.yoko-1):
                k = i*self.yoko + j
                std_diffx.append(std_w[k+1][0] - std_w[k][0])
        SFx = np.mean(std_diffx)    # 差の平均
        if SFx > 0:
            self.pn[0] = 1
        else:
            self.pn[0] = -1

        # Y軸方向
        std_diffy = []
        for i in range(self.tate-1):
            for j in range(self.yoko):
                k = i*self.yoko + j
                std_diffy.append(std_w[k + self.yoko][1] - std_w[k][1])
        SFy = np.mean(std_diffy)    # 差の平均
        if SFy > 0:
            self.pn[1] = 1
        else:
            self.pn[1] = -1

        # Z軸方向
        std_w_z = []
        for i in range(self.tate):
            for j in range(self.yoko):
                for k in range(stdnum_z+1):
                    stdpointsz, _ = cv2.projectPoints(np.float32([j,i,-k]), self.rvecs[-1], self.tvecs[-1], self.mtx, self.dist)
                    stdpoints2z, _ = cv2.projectPoints(np.float32([j,i,-k]), self.rvecs2[-1], self.tvecs2[-1], self.mtx2, self.dist2)
                    camera1_w, obj1_w = self.line_SEpoint(stdpointsz[0][0][0], stdpointsz[0][0][1], 1)
                    camera2_w, obj2_w = self.line_SEpoint(stdpoints2z[0][0][0], stdpoints2z[0][0][1], 2)
                    line1z = np.hstack((camera1_w[0].T, obj1_w[0].T)).reshape(2, 3)
                    line2z = np.hstack((camera2_w[0].T, obj2_w[0].T)).reshape(2, 3)
                    res_z = self.distance_2lines(line1z, line2z)
                    std_w_z.append(res_z[2])
        std_diffz = []
        for i in range(self.yoko*self.tate):
            for j in range(stdnum_z):
                k = j + i*(stdnum_z+1)
                std_diffz.append(std_w_z[k+1]-std_w_z[k])
        SFz = np.mean(std_diffz)
        if SFz > 0:
            self.pn[2] = 1
        else:
            self.pn[2] = -1

        self.SF = [SFx, SFy, SFz]


    def onMouse(self, event, x, y, flags, params):      # 1カメの画像に対するクリックイベント
        if event == cv2.EVENT_LBUTTONDOWN:                                              # 1カメ画像を左クリックしたら，
            self.obj1_i1x = x
            self.obj1_i1y = y
            self.camera1_w, self.obj1_w = self.line_SEpoint(x,y,1)                      # 1カメのワールド座標，1カメ画像でクリックされた点のワールド座標（この2点を通る直線のどこかにクリックしたものが存在することになる）
            self.click_count = 1                                                        # 1カメの画像をクリックしたことを伝える

    def onMouse2(self, event, x, y, flags, params):                                     # 2カメの画像に対するクリックイベント
        if event == cv2.EVENT_LBUTTONDOWN and self.click_count >= 1:                    # 1カメ画像が既にクリックされていて，2カメ画像をクリックしたら，
            self.obj2_i2x = x                                                           # 2カメの画像座標
            self.obj2_i2y = y
            self.camera2_w, self.obj2_w = self.line_SEpoint(x,y,2)                      # 2カメのワールド座標，2カメ画像でクリックされた点のワールド座標
            line1x = np.hstack((self.camera1_w[0].T, self.obj1_w[0].T)).reshape(2, 3)   # 1カメのワールド座標 と 1カメ画像でクリックされた点のワールド座標を通る直線
            line2x = np.hstack((self.camera2_w[0].T, self.obj2_w[0].T)).reshape(2, 3)   # 2カメのワールド座標 と 2カメ画像でクリックされた点のワールド座標を通る直線
            res = self.distance_2lines(line1x, line2x)                                  # 2本の直線の最接近点のワールド座標を求める

            result = [0,0,0]                                                            # 結果として出力するワールド座標値を定義
            result[0] = (self.pn[0] * (res[0])) / self.SF[0]             # 原点で引いて，スケールファクタで割る
            result[1] = (self.pn[1] * (res[1])) / self.SF[1] 
            result[2] = (self.pn[2] * (res[2])) / self.SF[2]
            
            result[0] = round(result[0], 4)                                             # 結果の値を四捨五入
            result[1] = round(result[1], 4)
            result[2] = round(result[2], 4)
            print(f'{result}\n')                                                        # 最終結果であるワールド座標を出力
            self.click_count = 2                                                        #  1カメ画像をクリックした後，2カメの画像をクリックしたことを伝える
            
    def line_SEpoint(self, x, y, num):      # 始点（カメラ）と終点（正規化画像座標）のワールド座標を求める関数，numは1カメか2カメか
        if num == 1:
            """
            undist_i1 = cv2.undistortPoints(np.float32([x,y]), self.mtx, self.dist, None, self.newcameramtx)
            obj_n1x = undist_i1[0][0][0]                                            # 対象物の1カメ画像座標
            obj_n1y = undist_i1[0][0][1]
            """
            obj_i1x = x                                                             # 対象物の1カメ画像座標
            obj_i1y = y

            obj_n1x = (obj_i1x - self.mtx[0][2]) / self.mtx[0][0]                   # 対象物の1カメ正規化座標　原点を真ん中にしてから，焦点距離で割る
            obj_n1y = (obj_i1y - self.mtx[1][2]) / self.mtx[1][1]
            
            obj_n1 = [[obj_n1x], [obj_n1y], [1]]                                    # 対象物の1カメ正規化画像座標系を1カメカメラ座標系に変換
            obj1_w = (np.linalg.inv(self.R)) @ (np.array(obj_n1) - np.array(self.tvecs))
            #obj1_w = (self.R.T) @ (np.array(obj_n1) - np.array(self.tvecs))                        # obj_n1を世界座標系に変換              Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
            
            #camera1_w = (np.linalg.inv(R)) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))     # 1カメのワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)
            camera1_w = (self.R.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))             # 1カメのワールド座標        Ｗ = Ｒ1^T (Ｃ1 - ｔ1)

            return camera1_w, obj1_w

        elif num == 2:
            """
            undist_i2 = cv2.undistortPoints(np.float32([x,y]), self.mtx2, self.dist2, None, self.newcameramtx2)
            obj_n2x = undist_i2[0][0][0]                                            # 対象物の2カメ画像座標
            obj_n2y = undist_i2[0][0][1]
            """
            obj_i2x = x                                                             # 対象物の1カメ画像座標
            obj_i2y = y

            obj_n2x = (obj_i2x - self.mtx2[0][2]) / self.mtx2[0][0]                 # 対象物の2カメ正規化座標　原点を真ん中にしてから，焦点距離で割る
            obj_n2y = (obj_i2y - self.mtx2[1][2]) / self.mtx2[1][1]
            
            obj_n2 = [[obj_n2x], [obj_n2y], [1]]                                    # 対象物の2カメ正規化画像座標系を2カメカメラ座標系に変換
            #obj1_w = (np.linalg.inv(R)) @ (np.array(obj_n1) - np.array(tvecs))
            obj2_w = (self.R2.T) @ (np.array(obj_n2) - np.array(self.tvecs2))                       # obj_n2を世界座標系に変換              Ｗ = Ｒ2^T (Ｃ2 - ｔ2)
            
            #camera2_w = (np.linalg.inv(R2)) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs2))        # 2カメのワールド座標        Ｗ = Ｒ2^T (Ｃ2 - ｔ2)
            camera2_w = (self.R2.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs2))           # 2カメのワールド座標        Ｗ = Ｒ2^T (Ｃ2 - ｔ2)

            return camera2_w, obj2_w

        return None, None


    def epipo(self,camera_w, obj_w, num):       
        if num == 1:            # 1カメ画像をクリックした点から2カメ画像にエピポーラ線を描画する場合
            camera1_c2 = np.array(self.R2) @ camera_w + np.array(self.tvecs2)                       # 2カメのカメラ座標系での1カメの位置
            camera1_i2 = self.mtx2 @ (camera1_c2/camera1_c2[0][2])                                  # 2カメの画像座標系での1カメの位置

            obj_c2 = np.array(self.R2) @ np.array(obj_w) + np.array(self.tvecs2)                    # obj_wを2カメのカメラ座標系に変換     Ｃ2 = Ｒ2Ｗ + ｔ2
            obj_i2 = self.mtx2 @ (obj_c2/obj_c2[0][2])                                              # obj_c2を2カメの画像座標に変換

            slope_i2 = (camera1_i2[0][1] - obj_i2[0][1])/(camera1_i2[0][0] - obj_i2[0][0])          # 線の傾き
            startpoint_i2y  = slope_i2*(0                    - obj_i2[0][0]) + obj_i2[0][1]         # エピポーラ線の2カメ画像の左端のy座標を求める
            endpoint_i2y   = slope_i2*(self.img_axes2.shape[1]   - obj_i2[0][0]) + obj_i2[0][1]     # エピポーラ線の2カメ画像の右端のy座標を求める

            return startpoint_i2y, endpoint_i2y

        elif num == 2:          # 2カメ画像をクリックした点から1カメ画像にエピポーラ線を描画する場合
            camera2_c1 = np.array(self.R) @ camera_w + np.array(self.tvecs)                         # 1カメのカメラ座標系での2カメの位置
            camera2_i1 = self.mtx @ (camera2_c1/camera2_c1[0][2])                                   # 1カメの画像座標系での2カメの位置

            obj_c1 = np.array(self.R) @ np.array(obj_w) + np.array(self.tvecs)                      # obj_wを1カメのカメラ座標系に変換     Ｃ1 = Ｒ1Ｗ + ｔ1
            obj_i1 = self.mtx @ (obj_c1/obj_c1[0][2])                                               # obj_c2を1カメの画像座標に変換

            slope_i1 = (camera2_i1[0][1] - obj_i1[0][1])/(camera2_i1[0][0] - obj_i1[0][0])          # 線の傾き
            startpoint_i1y  = slope_i1*(0                    - obj_i1[0][0]) + obj_i1[0][1]         # エピポーラ線の2カメ画像の左端のy座標を求める
            endpoint_i1y   = slope_i1*(self.img_axes2.shape[1]   - obj_i1[0][0]) + obj_i1[0][1]     # エピポーラ線の2カメ画像の右端のy座標を求める

            return startpoint_i1y, endpoint_i1y

        return None, None


    def distance_2lines(self, line1, line2):    # 直線同士の最接近距離と最接近点を求める関数
        '''
        直線同士の最接近距離と最接近点
        return 直線間の距離, line1上の最近接点、line2上の最近接点
        '''
        line1 = [np.array(line1[0]),np.array(line1[1])]
        line2 = [np.array(line2[0]),np.array(line2[1])]

        if abs(np.linalg.norm(line1[1]))<0.0000001:
            return None,None,None
        if abs(np.linalg.norm(line2[1]))<0.0000001:
            return None,None,None

        p1 = line1[0]
        p2 = line2[0]

        v1 = line1[1] / np.linalg.norm(line1[1])
        v2 = line2[1] / np.linalg.norm(line2[1])

        d1 = np.dot(p2 - p1,v1)
        d2 = np.dot(p2 - p1,v2)
        dv = np.dot(v1,v2)

        if (abs(abs(dv) - 1) < 0.0000001):
            v = np.cross(p2 - p1,v1)
            return np.linalg.norm(v),None,None

        t1 = (d1 - d2 * dv) / (1 - dv * dv)
        t2 = (d2 - d1 * dv) / (dv * dv - 1)

        #外挿を含む最近接点
        q1 = p1 + t1 * v1
        q2 = p2 + t2 * v2

        q1[0]=-q1[0]
        q1[1]=-q1[1]

        q2[0]=-q2[0]
        q2[1]=-q2[1]

        # XYZ座標の候補が2つあるため，平均をとる
        q3x = (q1[0]+q2[0])
        q3y = (q1[1]+q2[1])
        q3z = (q1[2]+q2[2])
        
        #return np.linalg.norm(q2 - q1), q1, q2
        return ([q3x, q3y, q3z])

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


def main():
    # 検出するチェッカーボードの交点の数
    tate = 3
    yoko = 4
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((tate*yoko,3), np.float32)
    objp[:,:2] = np.mgrid[0:yoko,0:tate].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    imgpoints2 = [] # 2d points in image plane.

    # 軸の定義
    axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3)

    val = None
    while True:
        val = input('Checkerboard or Prepared value? (C/P)')        # チェッカーボードを使うか用意した値を使うかを指定する
        if val == "c" or val == "C" or val == "p" or val == "P":    # CかPでなければ何度も聞く
            break

    if val == "c" or val == "C":            # チェッカーボードを使う場合
        img_axes0 = []
        imgpts0 = []
        corners02 = []

        img_axes0_2 = []
        imgpts0_2 = []
        corners02_2 = []

        ret01 = False
        ret02 = False
        
        while True:
            cap1 = cv2.VideoCapture(0)          #カメラの設定　デバイスIDは0
            _, frame1 = cap1.read()           #カメラからの画像取得

            if ret01 == True:
                img_axes0 = frame1.copy()
                img_axes0 = draw(img_axes0,corners02,imgpts0)
                cv2.imshow('camera1' , img_axes0)
            else:
                cv2.imshow('camera1' , frame1)
            cap1.release()
            

            cap2 = cv2.VideoCapture(2)          #カメラの設定　デバイスIDは0
            _, frame2 = cap2.read()           #カメラからの画像取得
            if ret02 == True:
                img_axes0_2 = frame2.copy()
                img_axes0_2 = draw(img_axes0_2,corners02_2,imgpts0_2)
                cv2.imshow('camera2' , img_axes0_2)
            else:
                cv2.imshow('camera2' , frame2)

            cap2.release()
            
            #繰り返し分から抜けるためのif文
            key =cv2.waitKey(1)
            if key == ord('s'):
                gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                _, img_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                # Find the chess board corners　交点を見つける
                ret, corners = cv2.findChessboardCorners(img_otsu, (yoko,tate),None)

                gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                _, img_otsu2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)
                # Find the chess board corners　交点を見つける
                ret2, corners2 = cv2.findChessboardCorners(img_otsu2, (yoko,tate),None)
                if ret == True and ret2 == True:
                    print("OK")
                    break
            elif key == ord('a'):
                ret01, corners02, imgpts0 = axes_check(frame1, tate, yoko, objp, criteria, axis)
                ret02, corners02_2, imgpts0_2 = axes_check(frame2, tate, yoko, objp, criteria, axis)
            elif key == 27:   #Escで終了
                break
        
        #cv2.imwrite('1.png',frame1)
        #cv2.imwrite('2.png',frame2)

        """
        frame1 = cv2.imread("1.png")
        frame2 = cv2.imread("2.png")
        cv2.imshow('camera1',frame1)
        cv2.imshow('camera2',frame2)
        """  
        
        gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        ret, img_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        ret2, img_otsu2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)

        # Find the chess board corners　交点を見つける
        #cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(img_otsu, (yoko,tate), None)
        ret2, corners2 = cv2.findChessboardCorners(img_otsu2, (yoko,tate),None)
        # If found, add object points, image points (after refining them)　交点が見つかったなら、描画
        corners12 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        corners22 = cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
        imgpoints.append(corners12)
        imgpoints2.append(corners22)

    elif val == "p" or val == "P":          # 用意した値を使う場合
        cap1 = cv2.VideoCapture(0)          #カメラの設定　デバイスIDは0
        _, frame1 = cap1.read()           #カメラからの画像取得
        cap1.release()
        cap2 = cv2.VideoCapture(2)
        _, frame2 = cap2.read()
        cap2.release()
        gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        gray2= cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        cv2.imshow('camera1' , frame1)
        cv2.imshow('camera2' , frame2)
        corners = np.array([254, 145, 289, 156, 327, 169, 368, 182, 412, 197, 459, 213, 224, 162, 262, 175, 300, 189, 342, 204, 387, 220, 436, 238, 192, 181, 229, 195, 270, 212, 313, 228, 359, 246, 410, 266, 156, 202, 194, 218, 235, 236, 280, 254, 329, 275, 381, 298, 115, 226, 155, 243, 197, 263, 243, 284, 293, 308, 349, 334],dtype='float32').reshape(-1,1,2)
        corners2 = np.array([125, 153, 180, 140, 235, 128, 290, 116, 342, 106, 392, 98, 131, 187, 191, 173, 250, 159, 308, 147, 364, 136, 416, 126, 140, 226, 204, 211, 268, 196, 329, 183, 388, 170, 443, 158, 151, 272, 220, 254, 288, 239, 354, 223, 416, 208, 472, 194, 164, 326, 238, 307, 312, 288, 382, 269, 447, 252, 505, 235],dtype='float32').reshape(-1,1,2)
        corners12 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        corners22 = cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
        imgpoints.append(corners12)
        imgpoints2.append(corners22)

    objpoints.append(objp)      # object point

    # パラメータの表示
    # Draw and display the corners
    #img = cv2.drawChessboardCorners(img, (yoko,tate), corners12,ret)
    #img2 = cv2.drawChessboardCorners(img2, (yoko,tate), corners22,ret2)
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
    
    # project 3D points to image plane
    imgpts, _ = cv2.projectPoints(axis, rvecs[-1], tvecs[-1], mtx, dist)
    imgpts2, _ = cv2.projectPoints(axis, rvecs2[-1], tvecs2[-1], mtx2, dist2)
    
    es = Estimation(mtx, dist, rvecs, tvecs, mtx2, dist2, rvecs2, tvecs2, frame2, imgpoints, imgpoints2, tate, yoko)

    cv2.setMouseCallback('camera1', es.onMouse)         # 1カメの画像に対するクリックイベント
    cv2.setMouseCallback('camera2', es.onMouse2)        # 2カメの画像に対するクリックイベント
    
    while True:
        cap1 = cv2.VideoCapture(0)          #カメラの設定　デバイスIDは0
        ret, frame1 = cap1.read()           #カメラからの画像取得
        img_axes = draw(frame1,corners12,imgpts)
        img_axes = es.line_update(img_axes,1)
        cv2.imshow('camera1', img_axes)      #カメラの画像の出力
        cap1.release()

        cap2 = cv2.VideoCapture(2)
        ret2, frame2 = cap2.read()
        img_axes2 = draw(frame2,corners22,imgpts2)
        img_axes2 = es.line_update(img_axes2,2)
        cv2.imshow('camera2', img_axes2)
        cap2.release()

        #繰り返し分から抜けるためのif文
        key =cv2.waitKey(1)
        if key == 27:   #Escで終了
            cv2.destroyAllWindows()
            break
        

        """
        while True:
            img_axes = draw(frame1,corners12,imgpts)
            cv2.imshow('camera1', img_axes)      #カメラの画像の出力

            frame22 = frame2.copy()
            img_axes2 = draw(frame22,corners22,imgpts2)
            img_axes2 = es.line_update(img_axes2)
            cv2.imshow('camera2', img_axes2)
            #繰り返し分から抜けるためのif文
            key =cv2.waitKey(1)
            if key == 27:   #Escで終了
                cv2.destroyAllWindows()
                break
        """


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