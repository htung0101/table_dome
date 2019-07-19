import os
import datetime

from utils.path import makedir
import ipdb
import threading
st = ipdb.set_trace




def run_rosrun(cam_id):
   os.system(f"rosrun ar_track_alvar individualMarkersNoKinect 5.6 0.08 0.2 /camera{cam_id}/color/image_raw /camera1/aligned_depth_to_color/camera_info /camera{cam_id}_color_optical_frame")

def run_vr_tag_save(cam_id, data_path):
   vr_tag_filename = os.path.join(data_path, "camera{cam_id}_vr_tag.txt")
   os.system(f"python ar_pose_listener.py --cam_no {cam_id} --out {vr_tag_filename}") 



"""
Get the position related to each camera for the vr tag

"""
data_root = "/home/zhouxian/data/TableDome/"


now = datetime.datetime.now()
# maybe add user name that will be cute
record_name = f"TableDome_y{now.year}_m{now.month}_h{now.hour}_m{now.minute}_s{now.second}"

data_path = os.path.join(data_root, record_name)
makedir(data_path)


num_cam = 1
for cam_id in range(num_cam):
    print("calibrate cam1")
    t1 = threading.Thread(target=run_rosrun, args=(cam_id,))
    t2 = threading.Thread(target=run_vr_tag_save, args=(cam_id, data_path,))
    t1.start()
    t2.start()

    t1.join(10)
    t2.join(10)




