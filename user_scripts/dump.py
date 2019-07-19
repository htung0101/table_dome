import os
import config
import multiprocessing
from utils.path import makedir
import ipdb
import time
st=ipdb.set_trace
def run_rgbd_writer(cam_id, save_dir, mode="depth"):
    max_nframes = int(config.FRAME_RATE * config.MAX_DURATION)
    os.system(f"python depthWrite.py --cam_no {cam_id} --save_dir {save_dir} --max_nframes {max_nframes} --mode {mode}")

def run_rosbag_play(rosbag_filename):
    time.sleep(10)
    os.system(f"rosbag play {rosbag_filename}")

def run_timestamp(cam_id, save_dir, mode="depth"):
    max_nframes = int(config.FRAME_RATE * config.MAX_DURATION)
    os.system(f"python timestamp.py --cam_no {cam_id} --save_dir {save_dir} --max_nframes {max_nframes} --mode {mode}")
    print("timestamp done")


record_root = os.path.join(config.data_root, config.record_name)
dump_folder = os.path.join(record_root, "rgb_depth_npy")
makedir(dump_folder)

dump_rgb_folder = os.path.join(dump_folder, "colorData")
makedir(dump_rgb_folder)

dump_depth_folder = os.path.join(dump_folder, "depthData")
makedir(dump_depth_folder)



bag_filename = os.path.join(record_root, "CalibData.bag")

num_cam = 1 #config.NUM_CAM

for cam_id in range(1, num_cam+1):
    
    
    # rgb
    dump_rgb_cam_folder = os.path.join(dump_rgb_folder, f"Cam{cam_id}")
    makedir(dump_rgb_cam_folder)
    
    process_rgbd_writer = multiprocessing.Process(target=run_rgbd_writer, args=(cam_id, dump_rgb_cam_folder, "rgb",))
    process_rosbag_play = multiprocessing.Process(target=run_rosbag_play, args=(bag_filename,))
    process_rgbd_writer.start()
    process_rosbag_play.start()

    process_rgbd_writer.join()
    process_rosbag_play.join()
    
 
    # depth
    dump_depth_cam_folder = os.path.join(dump_depth_folder, f"Cam{cam_id}")
    makedir(dump_depth_cam_folder)
    
    process_rgbd_writer = multiprocessing.Process(target=run_rgbd_writer, args=(cam_id, dump_depth_cam_folder, "depth",))
    process_rosbag_play = multiprocessing.Process(target=run_rosbag_play, args=(bag_filename,))
    process_rgbd_writer.start()
    process_rosbag_play.start()

    process_rgbd_writer.join()
    process_rosbag_play.join()
    
    
    
    #### timestamp
    # rgb
    
    process_timestamp = multiprocessing.Process(target=run_timestamp, args=(cam_id, dump_rgb_cam_folder, "rgb"))
    process_rosbag_play = multiprocessing.Process(target=run_rosbag_play, args=(bag_filename,))
    process_timestamp.start()
    process_rosbag_play.start()

    process_timestamp.join()
    process_rosbag_play.join()

    process_timestamp = multiprocessing.Process(target=run_timestamp, args=(cam_id, dump_depth_cam_folder, "depth" ))
    process_rosbag_play = multiprocessing.Process(target=run_rosbag_play, args=(bag_filename,))
    process_timestamp.start()
    process_rosbag_play.start()

    process_timestamp.join()
    process_rosbag_play.join()
    
    
    



    #t2.terminate()
    