import os
import sys
import datetime
import multiprocessing
import time

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


    bag_file_name = os.path.join(data_path, "CalibData.bag")
    os.system(f"rosbag record -O {bag_file_name} /camera1/color/image_raw_throttle_sync /camera1/aligned_depth_to_color/image_raw_throttle_sync /camera2/color/image_raw_throttle_sync /camera2/aligned_depth_to_color/image_raw_throttle_sync /camera3/color/image_raw_throttle_sync /camera3/aligned_depth_to_color/image_raw_throttle_sync /camera4/color/image_raw_throttle_sync /camera4/aligned_depth_to_color/image_raw_throttle_sync /camera5/color/image_raw_throttle_sync /camera5/aligned_depth_to_color/image_raw_throttle_sync /camera6/color/image_raw_throttle_sync /camera6/aligned_depth_to_color/image_raw_throttle_sync"
    )

def check_artag_folder(artag_folder, data_path):
    artag_folder_ = os.path.join(artag_folder, "ar_tag")
    if not os.path.exists(artag_folder_):
        print("ar_tag file does not exists, please check again: ", artag_folder_)
        sys.exit()

    for cam_id in range(1, config.NUM_CAM+1):
        if not os.path.isfile(os.path.join(artag_folder_, f"camera{cam_id}_ar_tag.pkl")):
            print("ar tag folder is invalid", f"camera{cam_id}_ar_tag.pkl does not exists")
            sys.exit()
    if not os.path.isfile(os.path.join(artag_folder_, "intrinsics.pkl")):
        print("ar tag folder is invalid", "intrinsics.pkl does not exists")
        sys.exit()

    if artag_folder != data_path:
        new_artag_folder = data_path
        makedir(new_artag_folder)
        os.system(f"scp -r {artag_folder_} {new_artag_folder}")

def get_record_name():

    # check the setting
    is_new_record_name = False
    record_name = config.record_name
    if config.artag_folder == "":
        print("artag_folder is empty in config file")
        print("Please do the calibration first or set artag_folder")
        sys.exit()
    elif config.record_name == "":
        is_new_record_name = True
        now = datetime.datetime.now()
        # maybe add user name that will be cute
        record_name = config.data_prefix + f"_TableDome_y{now.year}_m{now.month}_h{now.hour}_m{now.minute}_s{now.second}"
        data_path = os.path.join(config.data_root, record_name)
        makedir(data_path)
    return is_new_record_name, record_name

is_new_record_name, record_name = get_record_name()
data_path = os.path.join(config.data_root, record_name)
check_artag_folder(config.artag_folder, data_path)


all_process = []
num_cam = config.NUM_CAM


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

# check if the calibration is there and record the name

process_data_record = multiprocessing.Process(target=run_record_data, args=(data_path,))

all_process.append(process_data_record)

for process in all_process:
    process.start()

#time.sleep(t_start_record + config.duration)

#for process in all_process[::-1]:
#    process.terminate()
try:
    while(True):
        pass
except KeyboardInterrupt:
    if is_new_record_name:
        print("You have created a new record name,")
        print("please put your record name in the config file")
        print(f"record_name = \"{record_name}\"")





