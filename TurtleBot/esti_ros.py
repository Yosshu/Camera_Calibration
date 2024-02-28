#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from std_msgs.msg import Int16MultiArray

import numpy as np
import cv2
import glob
import pyautogui
import copy
import math
import colorsys
from numpy import linalg as LA
from itertools import product



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


def findSquare(img, bgr):                  # 指定したBGRの輪郭の中心の画像座標を取得する関数
    b = int(bgr[0])
    g = int(bgr[1])
    r = int(bgr[2])
    hsv = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    h = int(hsv[0]*180)
    s = int(hsv[1]*255)
    v = int(hsv[2]*255)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_color = np.array([h-10, s-60, v-120]) 
    upper_color = np.array([h+10, s+60, v+120]) 

    mask = cv2.inRange(img_hsv, lower_color, upper_color) 
    img_mask = cv2.bitwise_and(img, img, mask = mask)
    img_mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    _, img_mask_bin = cv2.threshold(img_mask_gray, 0, 255, cv2.THRESH_BINARY)
    #cv2.imshow('img_bin',img_mask_bin)

    contours = cv2.findContours(img_mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    # 面積が一定以上の輪郭のみ残す。
    area_thresh = 100
    contours = list(filter(lambda x: cv2.contourArea(x) > area_thresh, contours))
    ret = False
    center = 0
    centers = []
    # 輪郭を矩形で囲む。
    for cnt in contours:
        # 輪郭に外接する長方形を取得する。
        x, y, width, height = cv2.boundingRect(cnt)
        center = [x+(width/2),y+(height/2)]
        centers.append(center)
        # 描画する。
        cv2.rectangle(img, (x, y), (x + width, y + height), color=(255, g, r), thickness=2)
        ret = True
    centers = np.array(centers).reshape(-1,2)
    return ret, centers


def nearest(ret,a,b, du, dv):           # 赤と緑のペア探し
    if ret:
        na, nb = len(a), len(b)
        ## Combinations of a and b
        comb = product(range(na), range(nb))
        ## [[distance, index number(a), index number(b)], ... ]
        l = [[np.linalg.norm(a[ia] - b[ib]), ia, ib] for ia, ib in comb]
        ## Sort with distance
        l.sort(key=lambda x: x[0])
        _, ia, ib = l[0]
        # 求めた座標にdu, dvを足す
        res_a = a[ia] + np.array([du, dv])
        res_b = b[ib] + np.array([du, dv])
        return res_a, res_b 
    return None,None

def tangent_angle(u: np.ndarray, v: np.ndarray):        # 2つのベクトルのなす角を求める関数
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = i / n

    output = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))

    w = np.cross(u, v)
    if w < 0:
        output = -output

    return output


def isint(s):  # 整数値を表しているかどうかを判定
    try:
        int(s, 10)  # 文字列を実際にint関数で変換してみる
    except ValueError:
        return False  # 例外が発生＝変換できないのでFalseを返す
    else:
        return True  # 変換できたのでTrueを返す

def isfloat(s):  # 浮動小数点数値を表しているかどうかを判定
    try:
        float(s)  # 文字列を実際にfloat関数で変換してみる
    except ValueError:
        return False  # 例外が発生＝変換できないのでFalseを返す
    else:
        return True  # 変換できたのでTrueを返す



class Estimation:
    def __init__(self, mtx, dist, rvecs, tvecs, img, imgpoints, tate, yoko):
        self.mtx = mtx                  # 1カメの内部パラメータ
        self.dist = dist                # 1カメの歪み係数
        self.rvecs = rvecs              # 1カメの回転ベクトル
        self.tvecs = tvecs              # 1カメの並進ベクトル


        self.imgpoints = imgpoints      # 1カメで見つかったチェッカーボードの交点の画像座標
        self.tate = tate                # 検出するチェッカーボードの交点の縦の数
        self.yoko = yoko                # 検出するチェッカーボードの交点の横の数
        
        # 回転ベクトルを3×1から3×3に変換
        self.R, _ = cv2.Rodrigues(np.array(self.rvecs))     # 1カメの回転行列

        self.LRMclick = None    # 左右中クリックの判断

        # クラス内の関数間で共有したい変数
        self.obj1_i1x = 0               # 1カメでクリックした点の1カメ画像座標
        self.obj1_i1y = 0
        self.img = img      # 軸だけ描画された1カメの画像
        self.pn = [1, 1, 1]             # 1 or -1，ワールド座標がプラスかマイナスか，出力する直前にかける [X軸座標, Y軸座標, Z軸座標]
        #self.origin = []

        self.camera1_w = []             # 1カメのワールド座標
        self.obj1_w = []                # 1カメでクリックした点のワールド座標


        self.target_i = []  # 左or右クリックした点の画像座標
        self.target_w = []  # 左or右クリックした点のワールド座標

        self.scale = 50      # 1マス50cm

        self.robot_front_i = []
        self.robot_back_i = []
        self.robot_i = []

        self.robot_w = []
        self.robot_vector = []
        self.ret_robot = False
        self.input_angle = 0
        self.robotcam_height = 22.5
        self.robotcam_len = 16
        
        self.click = 0
        self.widthy = 1

        self.current_img = []
        self.RorG = "R"
        self.chosen_R_i = []
        self.chosen_G_i = []
        self.R_panel_color = []
        self.G_panel_color = []
        self.robot_approx_loc = []

        self.RANGE_RADIUS = 140



        self.depmtx = np.array([610.438, 0, 321.61958462, 
                                0, 610.827, 250.53954371, 
                                0, 0, 1]).reshape(3,3)                                                          # RealSenseの内部パラメータ
        self.depdist = np.array([0.06582571, 0.24423383, 0.00201921, 0.00146058, -1.27553243])                  # RealSenseの歪み係数
        #self.depf = (self.depmtx[0][0]+self.depmtx[1][1])/2     # RealSenseの焦点距離


        # ROS関連
        # Subscriberの作成
        self.sub = rospy.Subscriber('depth_data', Int16MultiArray, self.callback)
        # Publisherの作成
        self.pub = rospy.Publisher('toggle_wheel', Int16, queue_size=10)
    def callback(self,data):
        if self.ret_robot:
            datadata = data.data
            depu = datadata[0]
            depv = datadata[1]
            depth = datadata[2]

            #print(f'{depth} cm')
            robot_direction = np.arctan2(self.robot_vector[1],self.robot_vector[0])
            robot_cos = math.cos(robot_direction)
            robot_sin = math.sin(robot_direction)
            obj_w_x = (self.robot_w[0]*self.scale) + ((depth+self.robotcam_len) * robot_cos)
            obj_w_y = (self.robot_w[1]*self.scale) + ((depth+self.robotcam_len) * robot_sin)

            height = self.getHeight(depv,depth)

            width = self.getWidth(depu,depth)
            widthX = width*robot_sin
            widthY = width*robot_cos

            print(f'{[round(obj_w_x+widthX,2), round(obj_w_y-widthY,2), round(self.robotcam_height+height,2)]} [cm]')


    def getHeight(self,v,d):
        y = self.depmtx[1][2]-v
        h = (y*d)/self.depmtx[1][1]
        return h

    def getWidth(self,u,d):
        x = self.depmtx[0][2]-u
        w = (x*d)/self.depmtx[0][0]
        return w

    def talker(self, num):
        #rospy.loginfo(num)
        self.pub.publish(num)



    def onMouse(self, event, x, y, flags, params):      # 1カメの画像に対するクリックイベント
        if self.click == 0:
            if event == cv2.EVENT_LBUTTONDOWN:      # 左クリック
                self.click = 1
                """
                point_wx = input("Xw = ")
                point_wy = input("Yw = ")
                print()
                if isfloat(point_wx) and isfloat(point_wy):
                    point_wx = float(point_wx)/50
                    point_wy = float(point_wy)/50
                    self.target_w = [point_wx,point_wy,0.5]
                    pointw = np.float32([point_wx,point_wy,-0.5]).reshape(-1,3)
                    pointi, _ = cv2.projectPoints(pointw, self.rvecs[-1], self.tvecs[-1], self.mtx, self.dist)
                    self.target_i = [pointi[0][0][0],pointi[0][0][1]]
                else:"""
                self.target_i = [x,y]
                self.target_w = self.pointFixZ(x,y,0.5)
                self.LRMclick = 'L'
                self.click = 0
            elif event == cv2.EVENT_RBUTTONDOWN:    # 右クリック
                self.click = 1
                """
                angle = input("angle: ")
                print()
                if isfloat(angle):
                    self.input_angle = math.radians(float(angle))
                    self.LRMclick = 'R'
                else:"""
                self.target_i = [x,y]
                self.target_w = self.pointFixZ(x,y,0.5)
                self.LRMclick = 'R2'
                self.click = 0
                """elif event == cv2.EVENT_MBUTTONDOWN:    # 中クリック
                self.click = 1
                res = self.pointFixZ(x,y,0)
                res = [round(n*self.scale,2) for n in res]
                print(f"{res} [cm]")
                self.obj1_i1x = x
                self.obj1_i1y = y
                self.LRMclick = 'M'
                self.click = 0"""
            elif event == cv2.EVENT_MBUTTONDOWN:    # 中クリック
                self.click = 1
                if self.RorG == "R":
                    self.chosen_R_i = [x,y]
                    self.R_panel_color = list(self.current_img[y, x])
                    self.RorG = "G"
                elif self.RorG == "G":
                    self.chosen_G_i = [x,y]
                    self.G_panel_color = list(self.current_img[y, x])
                    self.RorG = "R"
                self.click = 0


                
    def find_robot_approx_loc(self):
        if self.chosen_R_i and self.chosen_G_i:
            if not self.robot_approx_loc:               # 赤緑を決めた最初
                self.middle_dot = [(self.chosen_R_i[0]+self.chosen_G_i[0])/2, (self.chosen_R_i[1]+self.chosen_G_i[1])/2]
            else:                                           # その後
                if self.robot_front_i is not None and self.robot_back_i is not None:        # ロボットが見つかっている場合
                    self.middle_dot = [(self.robot_front_i[0]+self.robot_back_i[0])/2, (self.robot_front_i[1]+self.robot_back_i[1])/2]
            range_u1 = self.middle_dot[0]-self.RANGE_RADIUS
            range_u2 = self.middle_dot[0]+self.RANGE_RADIUS
            range_v1 = self.middle_dot[1]-self.RANGE_RADIUS
            range_v2 = self.middle_dot[1]+self.RANGE_RADIUS
            self.robot_approx_loc = [[int(range_u1), int(range_v1)], [int(range_u2), int(range_v2)]]


    def crop_img_approx(self, img):
        if not self.robot_approx_loc:
            return img, [0, 0]
        approx_img = img[self.robot_approx_loc[0][1]:self.robot_approx_loc[1][1], self.robot_approx_loc[0][0]:self.robot_approx_loc[1][0]]
        return approx_img, self.robot_approx_loc[0]

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


    def line_update(self, img):        # エピポーラ線やクリックした点を描画する関数
        if self.LRMclick == 'L':
            img = cv2.circle(img, (int(self.target_i[0]),int(self.target_i[1])), 2, (0, 165, 255), thickness=-1)          # 左クリックした点を描画
        elif self.LRMclick == 'R2':
            img = cv2.circle(img, (int(self.target_i[0]),int(self.target_i[1])), 2, (255, 165, 0), thickness=-1)          # 右クリックした点を描画
        elif self.LRMclick == 'M':
            img = cv2.circle(img, (int(self.obj1_i1x),int(self.obj1_i1y)), 2, (255,0,255), thickness=-1)          # 中クリックした点を描画
        return img


    def undist_pts(self, pts_uv, num):
        if num == 1:
            pts_uv = cv2.undistortPoints(pts_uv, self.mtx, self.dist, P=self.mtx)
            pts_uv = pts_uv[0][0]
            pts_uv = [pts_uv[0],pts_uv[1]]
        return pts_uv

    def pointFixZ(self,ix,iy,iz):
        #floor_wz = 0
        obj_i = [ix,iy]
        floor_wz = -iz

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

        return [floor_wx,floor_wy,-floor_wz]

    """
    def getTarget(self):
        ret = False
        if self.Lclick_count == 1:
            ret = True
        return ret, self.target_i
    """

    def angleDiff(self,img,img2):        # ロボットの向きと目標方向の角度差，ロボットと目標位置の距離，左クリックしたのか右クリックしたのかを返す関数
        if not self.R_panel_color or not self.G_panel_color:
            return False,None,None,None
        approx_img, crop_pt1_i = self.crop_img_approx(img)
        ret1,reds_i= findSquare(approx_img, self.R_panel_color)       # 赤パネルの画像座標
        ret2,greens_i = findSquare(approx_img, self.G_panel_color)    # 緑パネルの画像座標
        ret3 = ret1 and ret2
        self.robot_front_i, self.robot_back_i = nearest(ret3, reds_i, greens_i, *crop_pt1_i)
        self.find_robot_approx_loc()
        if self.robot_approx_loc:
            # 四角形を描画
            img12 = img2.copy()
            cv2.rectangle(img12, (int(self.robot_approx_loc[0][0]), int(self.robot_approx_loc[0][1])), (int(self.robot_approx_loc[1][0]), int(self.robot_approx_loc[1][1])), (0, 255, 255), thickness=2)
            cv2.imshow("img12",img12)


        ret4 = False
        if ret3:
            if np.linalg.norm(self.robot_front_i-self.robot_back_i) < 50:
                ret4 = True
            else:
                ret4 = False

        if ret4:                                                                                             # ロボットが見つかったら
            robot_ix = (self.robot_front_i[0]+self.robot_back_i[0])/2
            robot_iy = (self.robot_front_i[1]+self.robot_back_i[1])/2
            self.robot_i = [robot_ix, robot_iy]
            self.ret_robot = True
            #robot_front_i = [(corners[0][0][0][0]+corners[0][0][1][0])/2,(corners[0][0][0][1]+corners[0][0][1][1])/2]
            #robot_back_i  = [(corners[0][0][2][0]+corners[0][0][3][0])/2,(corners[0][0][2][1]+corners[0][0][3][1])/2]
            robot_front_w = self.pointFixZ(self.robot_front_i[0],self.robot_front_i[1],0.5)       # 赤パネルのワールド座標
            robot_front_w_xy = np.array([robot_front_w[0],robot_front_w[1]])            # 赤パネルのワールド座標のXw,Yw
            robot_back_w = self.pointFixZ(self.robot_back_i[0],self.robot_back_i[1],0.5) # 緑パネルのワールド座標
            robot_back_w_xy = np.array([robot_back_w[0],robot_back_w[1]])      # 緑パネルのワールド座標のXw,Yw
            self.robot_vector = np.array(robot_front_w_xy - robot_back_w_xy)      # ロボットの向きベクトル
            robot_wx = (robot_front_w[0]+robot_back_w[0])/2                  # ロボットのワールド座標のXw
            robot_wy = (robot_front_w[1]+robot_back_w[1])/2                  # ロボットのワールド座標のYw
            self.robot_w = [robot_wx,robot_wy,0.5]
            robot_w_xy = np.array([robot_wx,robot_wy])          # ロボットのワールド座標のXw,Yw

            if (self.LRMclick == 'L' or self.LRMclick == 'R2'):                                              # 左クリックか右クリックしていたら
                target_w_xy = np.array([self.target_w[0],self.target_w[1]]) # 目標位置のワールド座標のXw,Yw
                target_vector = np.array(target_w_xy - robot_w_xy)  # 目的方向のベクトル
                angle = tangent_angle(self.robot_vector,target_vector)   # ロボットの向きと目標方向の角度差
                distance = math.sqrt((target_w_xy[0]-robot_w_xy[0])**2 + (target_w_xy[1]-robot_w_xy[1])**2)     # ロボットと目標位置の距離

                arrow_color = ()
                if self.LRMclick == 'L':
                    arrow_color = (0, 165, 255)
                elif self.LRMclick == 'R2':
                    arrow_color = (255, 165, 0)

                cv2.arrowedLine(img2,                                # 矢印
                    pt1=(int(self.robot_i[0]), int(self.robot_i[1])),
                    pt2=(int(self.target_i[0]), int(self.target_i[1])),
                    color=arrow_color,
                    thickness=3,
                    line_type=cv2.LINE_4,
                    shift=0,
                    tipLength=0.1)
                
                cv2.imshow("camera1",img2)
            
                return True,angle,distance,self.LRMclick
            elif self.LRMclick == 'R':
                horizontal_line = [1,0]
                target_vector_x = horizontal_line[0] * math.cos(self.input_angle) - horizontal_line[1] * math.sin(self.input_angle)
                target_vector_y = horizontal_line[0] * math.sin(self.input_angle) + horizontal_line[1] * math.cos(self.input_angle)
                target_vector = [target_vector_x, target_vector_y]
                angle = tangent_angle(self.robot_vector,target_vector)   # ロボットの向きと目標方向の角度差

                target_w = np.float32([robot_wx+target_vector_x,robot_wy+target_vector_y,-0.5]).reshape(-1,3)
                target_i, _ = cv2.projectPoints(target_w, self.rvecs[-1], self.tvecs[-1], self.mtx, self.dist)
                self.target_i = [target_i[0][0][0],target_i[0][0][1]]
                cv2.arrowedLine(img2,                                # 矢印
                    pt1=(int(self.robot_i[0]), int(self.robot_i[1])),
                    pt2=(int(self.target_i[0]), int(self.target_i[1])),
                    color=(255, 165, 0),
                    thickness=3,
                    line_type=cv2.LINE_4,
                    shift=0,
                    tipLength=0.1)
                cv2.imshow("camera1",img2)

                return True,angle,None,self.LRMclick
        else:
            self.ret_robot = False
        return False,None,None,None


    def find_robot_silhouette(self):
        if self.ret_robot:
            wid = 65
            robot_silhouette = [(int(self.robot_i[0]-wid), int(self.robot_i[1]-50)), (int(self.robot_i[0]+wid), int(self.robot_i[1]+100))]
            """
            img = self.img.copy()
            cv2.rectangle(img, robot_silhouette[0], robot_silhouette[1], (0, 0, 255), thickness=2)
            cv2.imshow("silhouette", img)
            """
            return robot_silhouette
        return 0




class ObstacleCandidatesFinder:
    def __init__(self, cap, es):
        self.es = es
        self.cap = cap
        _, self.frame = cap.read()
        # 背景モデルとクリックした色を保持するリストを初期化
        self.background_models = []
        self.clicked_colors = []

        # ウィンドウにマウスイベントを関連付ける
        cv2.namedWindow('Obstacle Candidates')
        cv2.setMouseCallback('Obstacle Candidates', self.on_mouse_click)

        # 遮蔽物検出座標を保持する変数
        self.floor_border = None

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # クリックした位置の色を取得
            clicked_color = self.frame[y, x]
            print("クリックした色:", clicked_color)

            # クリックした色を背景モデルに追加
            self.clicked_colors.append(clicked_color)

            # 背景モデルを更新（クリックした色ごとに異なる閾値を持つ）
            self.update_background_model()
        elif event == cv2.EVENT_MBUTTONDOWN:  # ホイールクリック
            # ホイールクリックで遮蔽物検出座標を設定
            self.floor_border = [(0, y), (self.frame.shape[1], y), (self.frame.shape[1], self.frame.shape[0]), (0, self.frame.shape[0])]

    def update_background_model(self):
        if self.clicked_colors:
            for color in self.clicked_colors:
                # クリックした色ごとに閾値を持つ背景モデルを作成
                model = {
                    'color': color,
                    'threshold': 30,  # 初期の閾値
                }
                self.background_models.append(model)
    
    def remove_noise(self, img):
        # ノイズ除去
        kernel_noise = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_noise)

        # より細かいノイズ除去
        kernel_detail = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_detail)

        # 膨張処理と収縮処理
        img = cv2.dilate(img, kernel_noise, iterations=3)
        img = cv2.erode(img, kernel_noise, iterations=3)

        return img

    def preprocess_floor_image(self, floor_image):
        # 白黒反転
        inverted_image = cv2.bitwise_not(floor_image)

        robot_silhouette = self.es.find_robot_silhouette()
        if robot_silhouette != 0:
            inverted_image[robot_silhouette[0][1]-self.floor_border[0][1]:robot_silhouette[1][1]-self.floor_border[0][1], robot_silhouette[0][0]:robot_silhouette[1][0]] = [0, 0, 0]  # BGR形式で黒色を指定

        # 白黒反転させた画像も表示
        #cv2.imshow('Inverted Image', inverted_image)

        # グレースケールに変換
        gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

        # ノイズ除去
        gray_image = self.remove_noise(gray_image)

        # 白い領域の輪郭を検出
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 一定サイズ以上の白い領域に対して四角形を描画
        min_contour_area = 1000  # 領域の最小面積
        obstacle_image = inverted_image.copy()
        candidate_rectangles = []  # 遮蔽物の候補となる四角形の外接矩形を保存するリスト
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                # 画像の端にある四角形は除外
                if x > 0 and y > 0 and x + w < floor_image.shape[1]: #and y + h < floor_image.shape[0]:
                    candidate_rectangles.append((x, y, x + w, y + h))

        # 外接矩形同士の包含関係をチェックし、完全に含まれている四角形を遮蔽物の候補から除外
        refined_candidate_rectangles = []  # 除外される四角形以外の四角形を保存するリスト
        for i in range(len(candidate_rectangles)):
            included = False
            x1, y1, x2, y2 = candidate_rectangles[i]
            for j in range(len(candidate_rectangles)):
                if i != j:
                    x3, y3, x4, y4 = candidate_rectangles[j]
                    if x1 >= x3 and y1 >= y3 and x2 <= x4 and y2 <= y4:
                        # 四角形iが四角形jに完全に含まれている場合、includedをTrueに設定して除外する
                        included = True
                        break
            if not included:
                refined_candidate_rectangles.append(candidate_rectangles[i])

        # 遮蔽物の候補の四角形を描画
        for rect in refined_candidate_rectangles:
            x1, y1, x2, y2 = rect
            cv2.rectangle(obstacle_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

        self.obstacle_candidates = refined_candidate_rectangles.copy()

        return obstacle_image

    def run(self):
        ret, self.frame = self.cap.read()
        if ret:
            if self.floor_border:
                # 遮蔽物検出座標が設定されている場合は、床を黒、遮蔽物を赤で表示
                floor_mask = np.zeros_like(self.frame, dtype=np.uint8)
                cv2.fillPoly(floor_mask, [np.array(self.floor_border)], color=(255, 255, 255))

                # クリップされた床領域に対して遮蔽物検出を行う
                if self.background_models:
                    combined_diff = np.zeros_like(self.frame)

                    for model in self.background_models:
                        # クリックした色ごとの背景差分を作成
                        mask = cv2.inRange(self.frame, model['color'] - model['threshold'], model['color'] + model['threshold'])
                        combined_diff = cv2.bitwise_or(combined_diff, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

                    result = cv2.bitwise_and(combined_diff, floor_mask)

                    # 床領域にトリミング
                    floor_image = result[self.floor_border[0][1]:self.floor_border[2][1], self.floor_border[0][0]:self.floor_border[1][0]]

                    # 床検出画像の前処理
                    processed_floor_image = self.preprocess_floor_image(floor_image)

                    cv2.imshow('Obstacle Detection', processed_floor_image)
                else:
                    # 背景モデルが存在しない場合は、オリジナル画像を表示
                    cv2.imshow('Obstacle Detection', self.frame)

            cv2.imshow('Obstacle Candidates', self.frame)




def main():
    # 検出するチェッカーボードの交点の数
    tate = 5
    yoko = 6
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((tate*yoko,3), np.float32)
    objp[:,:2] = np.mgrid[0:yoko,0:tate].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # 軸の定義
    axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)

    cap1 = cv2.VideoCapture(0)          #カメラの設定　デバイスIDは0
    # カメラの解像度を設定
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 幅の設定
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 高さの設定

    _, frame1 = cap1.read()           #カメラからの画像取得
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    cv2.imshow('camera1' , frame1)

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
    mtx = np.array([1.01644397e+03, 0.00000000e+00, 6.87903319e+02, 0.00000000e+00, 1.01960682e+03, 3.37505807e+02, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]).reshape(3,3)
    dist = np.array([-4.31940522e-01, 2.19409920e-01, -4.78545150e-05, 1.18269452e-03, -7.07910142e-02])

    corners = np.array([389, 267, 494, 264, 604, 263, 718, 262, 831, 261, 937, 261, 364, 338, 478, 337, 599, 335, 723, 335, 846, 334, 961, 332, 336, 424, 460, 424, 592, 424, 729, 424, 863, 420, 984, 417, 309, 523, 440, 528, 585, 531, 736, 531, 882, 525, 1011, 521, 279, 639, 420, 651, 577, 658, 743, 656, 904, 651, 1044, 636],dtype='float32').reshape(-1,1,2)
    corners12 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners12)

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
    
    angle_st = 5
    distance_st = 0.1

    ocf = ObstacleCandidatesFinder(cap1, es)

    rospy.init_node('Space')
    while True:
        talker_num = 0
        ret, frame1 = cap1.read()           #カメラからの画像取得
        es.current_img = frame1.copy()
        img_axes = frame1.copy()
        img_axes = draw(img_axes,corners12,imgpts)
        #frame1 = cv2.resize(frame1,dsize=(frame1.shape[1]*2,frame1.shape[0]*2))
        if ret:
            ocf.run()
            ####ocf.obstacle_candidates

            img_axes = es.line_update(img_axes)
            cv2.imshow('camera1', img_axes)      #カメラの画像の出力

            retd, angle, distance, LRM = es.angleDiff(frame1,img_axes)
            
            if retd:
                if -angle_st < angle < angle_st:        # 目的方向を向いていたら
                    if LRM == 'L':                      # 左クリックしていたら
                        if distance > distance_st:      # 目的地から離れていたら
                            talker_num = 3              # 前進
                        else:                           # 目的地に到着したら
                            talker_num = 0              # 停止
                            es.LRMclick = None
                    elif LRM == 'R' or 'R2':            # 右クリックしていたら
                        talker_num = 0                  # 停止
                        es.LRMclick = None
                elif angle_st <= angle:
                    talker_num = 1
                elif angle <= -angle_st:
                    talker_num = 2
            elif retd == False:                         # ロボットを見つけられなかったら
                talker_num = 0                          # 停止

        try:
            es.talker(talker_num)
        except rospy.ROSInterruptException: pass
            
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