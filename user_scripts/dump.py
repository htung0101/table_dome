import os
import config
import multiprocessing
from utils.path import makedir
import ipdb
st=ipdb.set_trace
def run_rgbd_writer(cam_id, save_dir):

    os.system("python depthWrite.py --cam_no {cam_id} --save_dir {save_dir}")

def run_rosbag_play(rosbag_filename):
    os.system(f"rosbag play {rosbag_filename}")


record_root = os.path.join(config.data_root, config.record_name)
dump_folder = os.path.join(record_root, "rgb_depth_npy")
makedir(dump_folder)

bag_filename = os.path.join(record_root, "CalibData.bag")

st()
num_cam = 1
for cam_id in range(1, num_cam+1):
    process_rgbd_writer = multiprocessing.Process(target=run_rgbd_writer, args=(cam_id, dump_folder, ))
    process_rosbag_play = multiprocessing.Process(target=run_rosbag_play, args=(bag_filename,))
    process_rgbd_writer.start()
    process_rosbag_play.start()
        #t2.start()
    #time.sleep(3)

    process_rgbd_writer.join()
    #t2.terminate()
    