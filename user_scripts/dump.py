import os
import config
import multiprocessing
from utils.path import makedir
import ipdb
import time
st=ipdb.set_trace
def run_rgbd_time_writer(cam_id, save_dir, mode="depth"):
    max_nframes = int(config.FRAME_RATE * config.MAX_DURATION)
    os.system(f"python rgbd_timestamp_write.py --cam_no {cam_id} --save_dir {save_dir} --max_nframes {max_nframes} --mode {mode}")

def run_rosbag_play(rosbag_filename):
    time.sleep(10)
    os.system(f"rosbag play {rosbag_filename}")

def run_intrinsic_writer(cam_id, save_dir):
    os.system(f"python intrinsics_write.py --cam_no {cam_id} --save_dir {save_dir}")

record_root = os.path.join(config.data_root, config.record_name)
dump_folder = os.path.join(record_root, "rgb_depth_npy")
makedir(dump_folder)

dump_rgb_folder = os.path.join(dump_folder, "colorData")
makedir(dump_rgb_folder)

dump_depth_folder = os.path.join(dump_folder, "depthData")
makedir(dump_depth_folder)



# dump rgbd, timestamp to folder
bag_filename = os.path.join(record_root, "CalibData.bag")


num_cam = 1 #config.NUM_CAM


for cam_id in range(1, num_cam+1):
    # get camera intrinsics

    dump_depth_cam_folder = os.path.join(dump_depth_folder, f"Cam{cam_id}")
    makedir(dump_depth_cam_folder)
    process_int_writer = multiprocessing.Process(target=run_intrinsic_writer, args=(cam_id, dump_depth_cam_folder,))
    process_rosbag_play = multiprocessing.Process(target=run_rosbag_play, args=(bag_filename,))

    process_int_writer.start()
    process_rosbag_play.start()

    process_int_writer.join()
    process_rosbag_play.join()
    
    """
    # rgb
    dump_rgb_cam_folder = os.path.join(dump_rgb_folder, f"Cam{cam_id}")
    makedir(dump_rgb_cam_folder)
    
    process_rgbd_writer1 = multiprocessing.Process(target=run_rgbd_time_writer, args=(cam_id, dump_rgb_cam_folder, "rgb",))
    process_rosbag_play = multiprocessing.Process(target=run_rosbag_play, args=(bag_filename,))

    process_rgbd_writer1.start()
    process_rosbag_play.start()

    process_rgbd_writer1.join()
    process_rosbag_play.join()
    # depth
    process_rgbd_writer2 = multiprocessing.Process(target=run_rgbd_time_writer, args=(cam_id, dump_depth_cam_folder, "depth",))
    process_rosbag_play = multiprocessing.Process(target=run_rosbag_play, args=(bag_filename,))

    process_rgbd_writer2.start()
    process_rosbag_play.start()

    process_rgbd_writer2.join()
    process_rosbag_play.join()
    """



"""
# build timestap compare
cmd = f"python timeCompare.py --save_dir {dump_folder}"
for cam_id in range(1, num_cam+1):
   dump_rgb_cam_folder = os.path.join(dump_rgb_folder, f"Cam{cam_id}")
   dump_depth_cam_folder = os.path.join(dump_depth_folder, f"Cam{cam_id}")

   cmd += f" --yaml_color{cam_id}Path "
   cmd += os.path.join(dump_rgb_cam_folder, "data.yaml")

   cmd += f" --yaml_depth{cam_id}Path "
   cmd += os.path.join(dump_depth_cam_folder, "data.yaml")

os.system(cmd)
"""

## write to tfrecord
#tfrecord_folder = os.path.join(record_root, "tfrecord")
#makedir(tfrecord_folder)
#os.system(f"python packtf.py --data_dir {record_root} --save_dir {tfrecord_folder}")
