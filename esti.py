from re import X
import numpy as np
import cv2
import glob
import pyautogui
import copy


# 3次元座標を描画
def draw(img, corners, imgpts):
    #corner = tuple(corners[0].ravel())
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (255,0,0), 5)   # x Blue
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0,255,0), 5)   # y Green
    img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 5)   # z Red
    #img = cv2.line(img, (int(corners[0][0][0]), int(corners[0][0][1])), (int(imgpts[3][0][0]), int(imgpts[3][0][1])), (255,0,255), 5)   # z Red
    return img

class Estimation:
    def __init__(self, mtx, dist, rvecs, tvecs, mtx2, dist2, rvecs2, tvecs2, img_axes2):
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.mtx2 = mtx2
        self.dist2 = dist2
        self.rvecs2 = rvecs2
        self.tvecs2 = tvecs2
        
        # 回転行列を3×1から3×3に変換
        self.R, _ = cv2.Rodrigues(np.array(self.rvecs))
        self.R2, _ = cv2.Rodrigues(np.array(self.rvecs2))

        self.click1 = 0     # 1枚目の画像をクリックしたか

        # 関数間で共有したい変数
        self.img_axes2 = img_axes2
        self.slope_i2 = 0
        self.img_line = []
        self.obj_i2 = []
        self.camera1_w = []
        self.obj_w = []


    # 画像のどこをクリックしたか返す
    def onMouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.Image1to2(x,y)

    def onMouse2(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and self.click1 == 1:
            self.Image2to1(x,y)
            
    def Image1to2(self, x, y):
        self.click1 = 1
        obj_i1x = x                                                             # 対象物の1カメ画像座標　クリックした点
        obj_i1y = y
        obj_n1x = (obj_i1x - self.mtx[0][2]) / self.mtx[0][0]                             # 対象物の1カメ正規化座標　原点を真ん中にしてから，焦点距離で割る
        obj_n1y = (obj_i1y - self.mtx[1][2]) / self.mtx[1][1]
        obj_n1 = [[obj_n1x], [obj_n1y], [1]]                                    # 対象物の1カメ正規化画像座標系を1カメカメラ座標系に変換
        #self.obj_w = (np.linalg.inv(R)) @ (np.array(obj_n1) - np.array(tvecs))
        self.obj_w = (self.R.T) @ (np.array(obj_n1) - np.array(self.tvecs))                    # obj_n1を世界座標系に変換              Ｗ = Ｒ^T (Ｃ1 - ｔ)
        obj_c2 = np.array(self.R2) @ np.array(self.obj_w) + np.array(self.tvecs2)              # obj_wを2カメのカメラ座標系に変換     Ｃ2 = ＲＷ + ｔ
        self.obj_i2 = self.mtx2 @ (obj_c2/obj_c2[0][2])                                   # obj_c2を2カメの画像座標に変換
        
        #self.camera1_w = (np.linalg.inv(R)) @ (np.array([[0], [0], [0]]) - np.array(tvecs))     # 1カメのワールド座標        Ｗ = Ｒ^T (Ｃ1 - ｔ)
        self.camera1_w = (self.R.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs))       # 1カメのワールド座標        Ｗ = Ｒ^T (Ｃ1 - ｔ)
        camera1_c2 = np.array(self.R2) @ self.camera1_w + np.array(self.tvecs2)                 # 2カメのカメラ座標系での1カメの位置
        camera1_i2 = self.mtx2 @ (camera1_c2/camera1_c2[0][2])                       # 2カメの画像座標系での1カメの位置

        self.img_line = self.img_axes2.copy()
        self.slope_i2 = (camera1_i2[0][1] - self.obj_i2[0][1])/(camera1_i2[0][0] - self.obj_i2[0][0])  # 傾き

        startpoint_i2y  = self.slope_i2*(0                    - self.obj_i2[0][0]) + self.obj_i2[0][1]
        goalpoint_i2y   = self.slope_i2*(self.img_axes2.shape[1]   - self.obj_i2[0][0]) + self.obj_i2[0][1]

        self.img_line = cv2.line(self.img_line, (0, int(startpoint_i2y)), (self.img_axes2.shape[1], int(goalpoint_i2y)), (0,255,255), 5)
        
        cv2.imshow('Axes2', self.img_line)


    def Image2to1(self, x, y):
        img_line2 = self.img_line.copy()
        option_y = self.slope_i2*(x - self.obj_i2[0][0][0]) + self.obj_i2[0][1][0]    # クリックされたx座標から線のy座標を求める
        option_y = option_y[0]  # 何故か配列になっているため[]をはずす
        if self.slope_i2 != 0:  # 線の傾きが0でないなら，
            option_x = (y - self.obj_i2[0][1][0])/self.slope_i2 + self.obj_i2[0][0][0]    # クリックされたy座標から線のx座標を求める
            option_x = option_x[0]  # 同じく何故か配列になっているため[]をはずす
            diff1 = abs(option_y - y)   # それぞれの差を求める
            diff2 = abs(option_x - x)
            if diff1 <= diff2:  # 差が小さい方を画像座標として採用する
                obj2_i2x = x
                obj2_i2y = option_y
            else:
                obj2_i2x = option_x
                obj2_i2y = y
        else:   # 線の傾きが0なら，y座標が求められないから，xとoption_yを画像座標として採用
            obj2_i2x = x
            obj2_i2y = option_y

        cv2.circle(img_line2, (int(obj2_i2x),int(obj2_i2y)), 8, (0, 165, 255), thickness=-1)
        cv2.imshow('Axes2',img_line2)
        obj2_n2x = (obj2_i2x - self.mtx2[0][2]) / self.mtx2[0][0]
        obj2_n2y = (obj2_i2y - self.mtx2[1][2]) / self.mtx2[1][1]
        obj2_n2 = [[obj2_n2x], [obj2_n2y], [1]]
        obj2_w = (np.array(self.R2.T)) @ (np.array(obj2_n2) - np.array(self.tvecs2))

        camera2_w = (self.R2.T) @ (np.array([[0], [0], [0]]) - np.array(self.tvecs2))       # 2カメのワールド座標        Ｗ = Ｒ^T (Ｃ2 - ｔ)

        line1 = np.hstack((self.camera1_w[0].T, self.obj_w[0].T)).reshape(2, 3)
        line2 = np.hstack((camera2_w[0].T, obj2_w[0].T)).reshape(2, 3)
        res = self.distance_2lines(line1,line2)
        print(res)
        print()


    def distance_2lines(self, line1, line2):
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
        #q1[2]=-q1[2]

        q2[0]=-q2[0]
        q2[1]=-q2[1]
        #q2[2]=-q2[2]

        # xyz座標の候補が2つあるため，平均をとる
        q3x = (q1[0]+q2[0])
        q3y = (q1[1]+q2[1])
        q3z = (q1[2]+q2[2])
        
        #return np.linalg.norm(q2 - q1), q1, q2
        return (q3x, q3y, q3z)



def main():
    # 検出するチェッカーボードの交点の数
    tate = 7
    yoko = 10
    pic_count = 0
    cap = cv2.VideoCapture(0)   #カメラの設定　デバイスIDは0
    while True:
        #カメラからの画像取得
        _, frame = cap.read()
        #カメラの画像の出力
        cv2.imshow('camera' , frame)
        key =cv2.waitKey(1)
        if key == ord('s'):
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            _, img_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            # Find the chess board corners　交点を見つける
            ret, corners = cv2.findChessboardCorners(img_otsu, (yoko,tate),None)
            if ret == True:
                pic_count += 1
                cv2.imwrite(str(pic_count) + '.png',frame)
                if pic_count == 2:
                    break
    #メモリを解放して終了するためのコマンド
    cap.release()
    cv2.destroyAllWindows()


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

        # 再投影誤差(Re-projection Error)
        # パラメータを評価する
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(objpoints)) )    # 0に近ければ近いほど良い

        # 軸の定義
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        #axis = np.float32([[3,0,0], [0,3,0], [0,0,-3], [6.5,4.5,0]]).reshape(-1,3)
        # project 3D points to image plane

        imgpts, jac = cv2.projectPoints(axis, rvecs[-1], tvecs[-1], mtx, dist)
        img_axes = draw(img_axes,corners12,imgpts)
        # img_axes = cv2.drawFrameAxes(img_axes, mtx, dist, rvecs[-1], tvecs[-1], 3, 3)
        cv2.imshow('Axes',img_axes)

        imgpts2, jac2 = cv2.projectPoints(axis, rvecs2[-1], tvecs2[-1], mtx2, dist2)
        img_axes2 = draw(img_axes2,corners22,imgpts2)
        # img_axes = cv2.drawFrameAxes(img_axes, mtx, dist, rvecs[-1], tvecs[-1], 3, 3)
        cv2.imshow('Axes2',img_axes2)
        
        es = Estimation(mtx, dist, rvecs, tvecs, mtx2, dist2, rvecs2, tvecs2, img_axes2)
        cv2.setMouseCallback('Axes', es.onMouse)
        cv2.setMouseCallback('Axes2', es.onMouse2)



    #繰り返し分から抜けるためのif文
    key =cv2.waitKey(0)
    if key == 27:   #Escで終了
        cv2.destroyAllWindows()



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
"""