import cv2
import numpy as np


class getImgpoints:
    def __init__(self):
        self.imgpoints1 = np.array([])
        
    def onMouse(self, event, x, y, flags, params):      # 1カメの画像に対するクリックイベント
        if event == cv2.EVENT_LBUTTONDOWN:          # 1カメ画像を左クリックしたら
            self.imgpoints1 = np.append(self.imgpoints1,(x,y)).reshape(-1,2)
        #if event == cv2.EVENT_RBUTTONDOWN and self.imgpoints1.size != 0:          # 1カメ画像を右クリックしたら
        #    self.imgpoints1 = np.delete(self.imgpoints1, -1, axis=0)

    def delete_imgpoints1(self):
        if self.imgpoints1.size != 0:
            self.imgpoints1 = np.delete(self.imgpoints1, -1, axis=0)
    
    def showPoints(self, img0, num):
        img = img0.copy()
        if num == 1:
            for i in self.imgpoints1:
                img = cv2.circle(img, (int(i[0]),int(i[1])), 2, (255, 0, 255), thickness=-1)
        return img

    def adjust(self, key):
        if self.imgpoints1.size != 0:
            if key == ord('w'):
                self.imgpoints1[-1][1] = self.imgpoints1[-1][1] - 1
            elif key == ord('a'):
                self.imgpoints1[-1][0] = self.imgpoints1[-1][0] - 1
            elif key == ord('s'):
                self.imgpoints1[-1][1] = self.imgpoints1[-1][1] + 1
            elif key == ord('d'):
                self.imgpoints1[-1][0] = self.imgpoints1[-1][0] + 1
    
    def printResult(self):
        show1 = []
        for i in self.imgpoints1:
            show1.append(int(i[0]))
            show1.append(int(i[1]))
        print(show1)
        


def main():
    gip = getImgpoints()
    cap1 = cv2.VideoCapture(0)          #カメラの設定　デバイスIDは0
    ret, frame1 = cap1.read()           #カメラからの画像取得
    flag = 0
    while True:
        if ret:
            img1 = gip.showPoints(frame1,1)
            cv2.imshow('camera1', img1)      #カメラの画像の出力
        if flag == 0:
            cv2.setMouseCallback('camera1', gip.onMouse)        # 1カメの画像に対するクリックイベント
            flag = 1
        key =cv2.waitKey(1)
        if key == ord('z'):
            gip.delete_imgpoints1()
        if key == 27:   #Escで終了
            cap1.release()
            cv2.destroyAllWindows()
            break
        gip.adjust(key)
        if key == ord('r'):
            ret, frame1 = cap1.read()           #カメラからの画像取得
            

    gip.printResult()
    #cv2.imwrite('1.png', frame1)


if __name__ == "__main__":
    main()