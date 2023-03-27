#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from std_msgs.msg import Int16MultiArray

import numpy as np
import cv2
aruco = cv2.aruco
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


def findSquare(img,b,g,r):                  # 指定したBGRの輪郭の中心の画像座標を取得する関数
    hsv = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    h = int(hsv[0]*180)
    s = int(hsv[1]*255)
    v = int(hsv[2]*255)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_color = np.array([h-25, s-60, v-120]) 
    upper_color = np.array([h+25, s+60, v+120]) 

    mask = cv2.inRange(img_hsv, lower_color, upper_color) 
    img_mask = cv2.bitwise_and(img, img, mask = mask)
    img_mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    _, img_mask_bin = cv2.threshold(img_mask_gray, 0, 255, cv2.THRESH_BINARY)
    #cv2.imshow('img_bin',img_mask_bin)

    contours = cv2.findContours(img_mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    # 面積が一定以上の輪郭のみ残す。
    area_thresh = 105
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
        #cv2.imshow('img_square',img)
        ret = True
    centers = np.array(centers).reshape(-1,2)
    return ret,centers


def nearest(ret,a,b):
    if ret:
        na, nb = len(a), len(b)
        ## Combinations of a and b
        comb = product(range(na), range(nb))
        ## [[distance, index number(a), index number(b)], ... ]
        l = [[np.linalg.norm(a[ia] - b[ib]), ia, ib] for ia, ib in comb]
        ## Sort with distance
        l.sort(key=lambda x: x[0])
        _, ia, ib = l[0]
        return a[ia], b[ib]
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

        self.robot_w = []
        self.robot_vector = []
        self.ret_robot = False
        self.input_angle = 0
        self.robotcam_height = 22.5
        self.robotcam_len = 16
        
        self.click = 0
        self.widthy = 1


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
            elif event == cv2.EVENT_MBUTTONDOWN:    # 中クリック
                self.click = 1
                res = self.pointFixZ(x,y,0)
                res = [round(n*self.scale,2) for n in res]
                print(f"{res} [cm]")
                self.obj1_i1x = x
                self.obj1_i1y = y
                self.LRMclick = 'M'
                self.click = 0


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

    def angleDiff(self,img,img2,dictionary,parameters):        # ロボットの向きと目標方向の角度差，ロボットと目標位置の距離，左クリックしたのか右クリックしたのかを返す関数
        #corners, ids, _ = aruco.detectMarkers(img, dictionary, parameters=parameters)
        ret1,reds_i= findSquare(img,57,67,255)       # 赤パネルの画像座標
        ret2,greens_i = findSquare(img,98,142,53)    # 緑パネルの画像座標
        ret3 = ret1 and ret2
        robot_front_i,robot_back_i = nearest(ret3,reds_i,greens_i)
        ret4 = False
        if ret3:
            if np.linalg.norm(robot_front_i-robot_back_i) < 50:
                ret4 = True
            else:
                ret4 = False


        if ret4:                                                                                             # ロボットが見つかったら
            self.ret_robot = True
            #robot_front_i = [(corners[0][0][0][0]+corners[0][0][1][0])/2,(corners[0][0][0][1]+corners[0][0][1][1])/2]
            #robot_back_i  = [(corners[0][0][2][0]+corners[0][0][3][0])/2,(corners[0][0][2][1]+corners[0][0][3][1])/2]
            robot_front_w = self.pointFixZ(robot_front_i[0],robot_front_i[1],0.5)       # 赤パネルのワールド座標
            robot_front_w_xy = np.array([robot_front_w[0],robot_front_w[1]])            # 赤パネルのワールド座標のXw,Yw
            robot_back_w = self.pointFixZ(robot_back_i[0],robot_back_i[1],0.5) # 緑パネルのワールド座標
            robot_back_w_xy = np.array([robot_back_w[0],robot_back_w[1]])      # 緑パネルのワールド座標のXw,Yw
            self.robot_vector = np.array(robot_front_w_xy - robot_back_w_xy)      # ロボットの向きベクトル
            robot_wx = (robot_front_w[0]+robot_back_w[0])/2                  # ロボットのワールド座標のXw
            robot_wy = (robot_front_w[1]+robot_back_w[1])/2                  # ロボットのワールド座標のYw
            self.robot_w = [robot_wx,robot_wy,0.5]
            robot_w_xy = np.array([robot_wx,robot_wy])          # ロボットのワールド座標のXw,Yw

            robot_ix = (robot_front_i[0]+robot_back_i[0])/2
            robot_iy = (robot_front_i[1]+robot_back_i[1])/2
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
                    pt1=(int(robot_ix), int(robot_iy)),
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
                #print([-robot_ix+target_vector_x,-robot_iy+target_vector_y,-0.5])

                target_w = np.float32([robot_wx+target_vector_x,robot_wy+target_vector_y,-0.5]).reshape(-1,3)
                target_i, _ = cv2.projectPoints(target_w, self.rvecs[-1], self.tvecs[-1], self.mtx, self.dist)
                self.target_i = [target_i[0][0][0],target_i[0][0][1]]
                cv2.arrowedLine(img2,                                # 矢印
                    pt1=(int(robot_ix), int(robot_iy)),
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
    axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3)


    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters =  aruco.DetectorParameters_create()
    # CORNER_REFINE_NONE, no refinement. CORNER_REFINE_SUBPIX, do subpixel refinement. CORNER_REFINE_CONTOUR use contour-Points
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

    cap1 = cv2.VideoCapture(0)          #カメラの設定　デバイスIDは0



    _, frame1 = cap1.read()           #カメラからの画像取得
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    cv2.imshow('camera1' , frame1)
    corners = np.array([138, 178, 209, 179, 283, 180, 359, 183, 434, 185, 506, 188, 121, 226, 196, 228, 278, 230, 361, 233, 444, 235, 522, 236, 100, 283, 182, 287, 272, 290, 364, 292, 454, 294, 538, 294, 79, 350, 167, 358, 265, 363, 366, 365, 465, 365, 556, 363, 57, 426, 152, 439, 257, 447, 368, 451, 475, 449, 575, 442],dtype='float32').reshape(-1,1,2)
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
    mtx = np.array([679.96806792, 0, 346.17428752, 0, 680.073095, 223.11864352, 0, 0, 1]).reshape(3,3)
    # 640,480

    dist = np.array([-4.65322789e-01, 3.88192556e-01, -2.58061417e-03, -1.69216076e-04, -3.97886097e-01])

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

    rospy.init_node('Space')
    while True:
        talker_num = 0
        ret, frame1 = cap1.read()           #カメラからの画像取得
        img_axes = frame1.copy()
        img_axes = draw(img_axes,corners12,imgpts)
        #frame1 = cv2.resize(frame1,dsize=(frame1.shape[1]*2,frame1.shape[0]*2))
        if ret:
            img_axes = es.line_update(img_axes)
            cv2.imshow('camera1', img_axes)      #カメラの画像の出力

            retd, angle, distance, LRM = es.angleDiff(frame1,img_axes,dictionary,parameters)
            if retd:
                #print(angle)
                if -angle_st < angle < angle_st:          # 目的方向を向いていたら
                    if LRM == 'L':                  # 左クリックしていたら
                        if distance > distance_st:              # 目的地から離れていたら
                            talker_num = 3               # 前進
                        else:                           # 目的地に到着したら
                            talker_num = 0               # 停止
                    elif LRM == 'R' or 'R2':                # 右クリックしていたら
                        talker_num = 0
                elif angle_st <= angle:
                    talker_num = 1
                elif angle <= -angle_st:
                    talker_num = 2
            elif retd == False:
                talker_num = 0

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