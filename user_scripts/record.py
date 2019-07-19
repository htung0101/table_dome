import os
import multiprocessing
import config
import time
import ipdb
st=ipdb.set_trace
# start the camera by running roslaunch


t_launch_cam = 15
def run_launch_cameras():
    os.system("source /home/zhouxian/catkin_ws/devel/setup.bash")
    os.system("roslaunch realsense2_camera rs_aligned_depth_multiple_cameras.launch camera1:=camera1 serial_nocamera1:=836612072253 camera2:=camera2 serial_nocamera2:=838212071161 camera3:=camera3 serial_nocamera3:=838212071165 camera4:=camera4 serial_nocamera4:=831612072676 camera5:=camera5 serial_nocamera5:=826212070528 camera6:=camera6 serial_nocamera6:=838212071158")


def run_4hz(cam_id):
    time.sleep(t_launch_cam )
    print(f"reduce frame rate for camera {cam_id}")
    os.system(f"rosrun topic_tools throttle messages /camera{cam_id}/color/image_raw 4.0")

def run_data_sync():
    time.sleep(t_launch_cam + 10)
    print(f"data sync...")
    os.system("python dataSync.py")


def run_record_data(data_path):
    time.sleep(t_launch_cam + 20)
    print(f"=====================START RECORDING in 5 secs===================")
    for t in range(5)[::-1]:
        print("count down:", t)
        time.sleep(1)
    print(f"=====================START RECORDING=====================")


    bag_file_name = os.path.join(data_path, "CalibData.bag")
    os.system(f"rosbag record -O {bag_file_name} /camera1/color/image_raw_throttle_sync /camera1/aligned_depth_to_color/image_raw_throttle_sync /camera2/color/image_raw_throttle_sync /camera2/aligned_depth_to_color/image_raw_throttle_sync /camera3/color/image_raw_throttle_sync /camera3/aligned_depth_to_color/image_raw_throttle_sync /camera4/color/image_raw_throttle_sync /camera4/aligned_depth_to_color/image_raw_throttle_sync /camera5/color/image_raw_throttle_sync /camera5/aligned_depth_to_color/image_raw_throttle_sync /camera6/color/image_raw_throttle_sync /camera6/aligned_depth_to_color/image_raw_throttle_sync"
    )

all_process = []
num_cam = config.NUM_CAM


process_cam_launch = multiprocessing.Process(target=run_launch_cameras, args=())
all_process.append(process_cam_launch)

processes_cam_hertz = []
for cam_id in range(1, num_cam+1):
    processes_cam_hertz.append(
        multiprocessing.Process(target=run_4hz, args=(cam_id,)))
all_process += processes_cam_hertz

process_data_sync = multiprocessing.Process(target=run_data_sync, args=())
all_process.append(process_data_sync)

data_path = os.path.join(config.data_root, config.record_name)
process_data_record = multiprocessing.Process(target=run_record_data, args=(data_path,))

all_process.append(process_data_record)

for process in all_process:
    process.start()




