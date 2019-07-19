
import os

# start the camera by running roslaunch
os.system("source /home/zhouxian/catkin_ws/devel/setup.bash")

os.system("roslaunch realsense2_camera rs_aligned_depth_multiple_cameras.launch camera1:=camera1 serial_nocamera1:=836612072253 camera2:=camera2 serial_nocamera2:=838212071161 camera3:=camera3 serial_nocamera3:=838212071165 camera4:=camera4 serial_nocamera4:=831612072676 camera5:=camera5 serial_nocamera5:=826212070528 camera6:=camera6 serial_nocamera6:=838212071158")



os.system("python dataSync.py")