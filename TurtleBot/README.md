# ROSで制御
1. roscore実行
2. Arduinoでシリアルポート指定
3. rosrun rosserial_python serial_node.py /dev/ttyACM0
4. rostopic pub toggle_wheel std_msgs/Empty --once

