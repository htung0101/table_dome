import os
import sys
import datetime
import multiprocessing
import time
import timeit
import config
from utils.path import makedir

import ipdb
st=ipdb.set_trace
# start the camera by running roslaunch


t_launch_cam = 15
t_4hz = 10
t_data_sync = 10
t_start_record = t_launch_cam + t_4hz + t_data_sync + 5


def run_launch_cameras():
    os.system("source /home/zhouxian/catkin_ws/devel/setup.bash")
    os.system("roslaunch realsense2_camera rs_aligned_depth_multiple_cameras.launch camera1:=camera1 serial_nocamera1:=836612072253 camera2:=camera2 serial_nocamera2:=838212071161 camera3:=camera3 serial_nocamera3:=838212071165 camera4:=camera4 serial_nocamera4:=831612072676 camera5:=camera5 serial_nocamera5:=826212070528 camera6:=camera6 serial_nocamera6:=838212071158")


def run_4hz_rgb(cam_id):
    time.sleep(t_launch_cam)
    print(f"reduce frame rate for camera {cam_id}")
    os.system(f"rosrun topic_tools throttle messages /camera{cam_id}/color/image_raw {config.FRAME_RATE}")

def run_4hz_depth(cam_id):
    time.sleep(t_launch_cam)
    print(f"reduce frame rate for camera {cam_id}")
    os.system(f"rosrun topic_tools throttle messages /camera{cam_id}/aligned_depth_to_color/image_raw 4.0")


def run_data_sync():
    time.sleep(t_launch_cam + t_4hz)
    print(f"data sync...")
    os.system("python dataSync.py")


def run_record_data(data_path):

    time.sleep(t_launch_cam + t_4hz + t_data_sync)
    print(f"=====================START RECORDING in 5 secs===================")
    for t in range(5)[::-1]:
        print("count down:", t)
        time.sleep(1)
    print(f"=====================START RECORDING=====================")

    bag_file_name = os.path.join(data_path, "Checkerboard_CalibData.bag")
    os.system(f"rosbag record -O {bag_file_name} /camera1/color/image_raw_throttle_sync /camera1/aligned_depth_to_color/image_raw_throttle_sync /camera2/color/image_raw_throttle_sync /camera2/aligned_depth_to_color/image_raw_throttle_sync /camera3/color/image_raw_throttle_sync /camera3/aligned_depth_to_color/image_raw_throttle_sync /camera4/color/image_raw_throttle_sync /camera4/aligned_depth_to_color/image_raw_throttle_sync /camera5/color/image_raw_throttle_sync /camera5/aligned_depth_to_color/image_raw_throttle_sync /camera6/color/image_raw_throttle_sync /camera6/aligned_depth_to_color/image_raw_throttle_sync"
    )

all_process = []
num_cam = config.NUM_CAM

if config.checkerboard_record_name == "": # if it is empty, then generate a new one
    now = datetime.datetime.now()
    checkerboard_record_name = f"TableDome_y{now.year}_m{now.month}_h{now.hour}_m{now.minute}_s{now.second}"
    data_path = os.path.join(config.checkerboard_data_root, checkerboard_record_name)
    makedir(data_path)

else:
    data_path = os.path.join(config.checkerboard_data_root, config.checkerboard_record_name)
    # check if the file is there
    if not os.path.exists(data_path):
        print(f"try to load from {data_path} but it does not exist.")

    else:
        print(f"checkboard file is already there. You don't need to run this step.")
    print("If you don't want to load from file, set checkboard_record_name in config.py to empty string")
    print("checkboard_record_name=\"\"")
    sys.exit()

#process_cam_launch = multiprocessing.Process(target=run_launch_cameras, args=())
#all_process.append(process_cam_launch)

processes_cam_hertz = []
for cam_id in range(1, num_cam+1):
    processes_cam_hertz.append(
        multiprocessing.Process(target=run_4hz_rgb, args=(cam_id,)))
    processes_cam_hertz.append(
        multiprocessing.Process(target=run_4hz_depth, args=(cam_id,)))
all_process += processes_cam_hertz

process_data_sync = multiprocessing.Process(target=run_data_sync, args=())
all_process.append(process_data_sync)

process_data_record = multiprocessing.Process(target=run_record_data, args=(data_path,))
all_process.append(process_data_record)

for process in all_process:
    process.start()




# it roughly takes 40 secs to set up and record for 2 minutes
time.sleep(40 + 60)


#for process in all_process:
#    process.terminate()


print("Stop at any time you want!")

try:
    while(True):
        print("Stop at any time you want!")
except KeyboardInterrupt:

 if config.checkerboard_record_name == "":
    print("please put your record name in the config file")
    print(f"checkboard_record_name = \"{checkerboard_record_name}\"")
    print("If you no longer want it, please delete it with")
    print(f"rm -rf {data_path}")
    print("And remember to put away the vr tag before you start recording!")

    print("next, run the optimization with")
    bag_filename = os.path.join(data_path, "Checkerboard_CalibData.bag")
    print(f"kalibr_calibrate_cameras --target april_6x6.yaml --bag {bag_filename} --models pinhole-equi pinhole-equi pinhole-equi pinhole-equi pinhole-equi pinhole-equi --topics /camera1/color/image_raw_throttle_sync /camera2/color/image_raw_throttle_sync /camera3/color/image_raw_throttle_sync /camera4/color/image_raw_throttle_sync /camera5/color/image_raw_throttle_sync /camera6/color/image_raw_throttle_sync")
