import os
import sys
import datetime

from utils.path import makedir
import ipdb
#import threading
import multiprocessing
import time
import subprocess
import config

st = ipdb.set_trace


def run_launch_cameras():
    os.system("source /home/zhouxian/catkin_ws/devel/setup.bash")
    os.system("roslaunch realsense2_camera rs_aligned_depth_multiple_cameras.launch camera1:=camera1 serial_nocamera1:=836612072253 camera2:=camera2 serial_nocamera2:=838212071161 camera3:=camera3 serial_nocamera3:=838212071165 camera4:=camera4 serial_nocamera4:=831612072676 camera5:=camera5 serial_nocamera5:=826212070528 camera6:=camera6 serial_nocamera6:=838212071158")



def run_rosrun(cam_id):
   os.system(f"rosrun ar_track_alvar individualMarkersNoKinect 5.6 0.08 0.2 /camera{cam_id}/color/image_raw /camera1/aligned_depth_to_color/camera_info /camera{cam_id}_color_optical_frame")

def run_ar_tag_save(cam_id, data_path):
   time.sleep(2)
   ar_tag_filename = os.path.join(data_path, f"camera{cam_id}_ar_tag.pkl")
   print(ar_tag_filename)
   os.system(f"python ar_pose_listener.py --cam_no {cam_id} --out {ar_tag_filename}")


def after_timeout(): 
    print("KILL MAIN THREAD: %s" % threading.currentThread().ident) 
    raise SystemExit 

def run_intrinsic_writer(num_cam, save_dir):
    os.system(f"python intrinsics_write_all.py --num_cam {num_cam} --save_dir {save_dir}")


"""
Get the position related to each camera for the vr tag

"""
#data_root = "/home/zhouxian/data/TableDome/"

data_root = config.data_root
num_cam = config.NUM_CAM

# check if artag folder is empty

#if os.path.exists(config.artag_folder):
if config.artag_folder is not "":
   print("artag folder exists: config.artag_folder")
   print("so we don't calibrate again.")
   print("If you really want to calibrate,")
   print("please set it to empty in config.py: artag_folder=\"\" ")
   sys.exit()

now = datetime.datetime.now()
# maybe add user name that will be cute
record_name = config.data_prefix + f"_TableDome_y{now.year}_m{now.month}_h{now.hour}_m{now.minute}_s{now.second}"

data_path = os.path.join(data_root, record_name)
makedir(data_path)
print("===========================================")
print("You are recroding things into:", data_path)
print("===========================================")
#t0 = multiprocessing.Process(target=run_launch_cameras, args=())
#t0.start()
# colleting information for the cameras
ar_tag_data_path = os.path.join(data_path, "ar_tag")
makedir(ar_tag_data_path)
for cam_id in range(1, num_cam+1):
    print(f"####################calibrate cam {cam_id}##################")
    t1 = multiprocessing.Process(target=run_rosrun, args=(cam_id,))
    t2 = multiprocessing.Process(target=run_ar_tag_save, args=(cam_id, ar_tag_data_path,))
    t1.start()
    t2.start()
    time.sleep(5)

    t1.terminate()
    t2.terminate()

print("################## get intrinsics ############################3")
t1 = multiprocessing.Process(target=run_intrinsic_writer, args=(num_cam, ar_tag_data_path))

t1.start()
time.sleep(3)

t1.terminate()

#t0.terminate()
artag_folder = os.path.join(os.path.join(data_root, record_name))

print("please put your record name in the config file")
print(f"record_name = \"{record_name}\"")
print("and set your artag folder to")
print(f"artag_folder = \"{artag_folder}\"")
print("")
print("If you no longer want it, please delete it with")
print(f"rm -rf {data_path}")
print("And remember to put away the ar tag before you start recording!")
