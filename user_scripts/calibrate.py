import os
import datetime

from utils.path import makedir
import ipdb
#import threading
import multiprocessing
import time
import subprocess
st = ipdb.set_trace




def run_rosrun(cam_id):
   os.system(f"rosrun ar_track_alvar individualMarkersNoKinect 5.6 0.08 0.2 /camera{cam_id}/color/image_raw /camera1/aligned_depth_to_color/camera_info /camera{cam_id}_color_optical_frame")

def run_vr_tag_save(cam_id, data_path):
   vr_tag_filename = os.path.join(data_path, f"camera{cam_id}_vr_tag.pkl")
   print(vr_tag_filename)
   os.system(f"python ar_pose_listener.py --cam_no {cam_id} --out {vr_tag_filename}") 


def after_timeout(): 
    print("KILL MAIN THREAD: %s" % threading.currentThread().ident) 
    raise SystemExit 



"""
Get the position related to each camera for the vr tag

"""
data_root = "/home/zhouxian/data/TableDome/"


now = datetime.datetime.now()
# maybe add user name that will be cute
record_name = f"TableDome_y{now.year}_m{now.month}_h{now.hour}_m{now.minute}_s{now.second}"

data_path = os.path.join(data_root, record_name)
makedir(data_path)
print("===========================================")
print("You are recroding things into:", data_path)
print("===========================================")

# colleting information for the cameras
vr_tag_data_path = os.path.join(data_path, "vr_tag")
makedir(vr_tag_data_path)
num_cam = 6
for cam_id in range(1, num_cam+1):
    print("####################calibrate cam {cam_id}##################")
    t1 = multiprocessing.Process(target=run_rosrun, args=(cam_id,))
    t2 = multiprocessing.Process(target=run_vr_tag_save, args=(cam_id, vr_tag_data_path,))
    t1.start()
    t2.start()
    time.sleep(3)

    t1.terminate()
    t2.terminate()

os.system("merge_vr_cam.py --data_path data_root  --record_name record_name --num_cam num_cam")





