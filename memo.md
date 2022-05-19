# カメラキャリブレーションパラメータ(Camera Calibration Parameters)

## 外部パラメータ(extrinsic parameters)
　外部パラメータとは，世界座標をカメラ座標に変換する行列のこと．  
　世界座標系(ワールド座標系，グローバル座標系)とは，3次元空間における絶対的な位置のこと．一般的には，現実世界での緯度・経度・高度のようなものを指す．
ただし，カメラキャリブレーションにおける世界座標系は，床やチェッカーボードの端などを原点とする．  
　カメラ座標系とは，カメラを原点とした時の座標のこと．つまり，カメラから見て，物体がどの位置にあるかを表すことができる．

## 内部パラメータ(intrinsic parameters)
　内部パラメータとは，カメラ座標を画像座標へ変換する行列のこと．3次元座標から2次元座標に射影変換する．焦点距離(focal length)，光学的中心(optical center)(主点 principal pointともいう)，せん断係数(skew coefficient)が含まれる．  
　画像座標系とは，カメラに映った画像における位置を表す2次元座標のこと．


## 【参考】
・カメラ キャリブレーションとは - MATLAB & Simulink - MathWorks 日本　(https://jp.mathworks.com/help/vision/ug/camera-calibration.html)  
・カメラ外部パラメータとは | NO MORE! 車輪の再発明　(https://mem-archive.com/2018/02/17/post-74/)  
・カメラ内部パラメータとは | NO MORE! 車輪の再発明　(https://mem-archive.com/2018/02/21/post-157/)  


