import os
import config
import multiprocessing
from utils.path import makedir
import ipdb
import time
import sys
st=ipdb.set_trace
def run_rgbd_time_writer(cam_id, save_dir, mode="depth"):
    max_nframes = int(config.FRAME_RATE * config.MAX_DURATION)
    node_suffix = f'_{cam_id}_{mode}'
    os.system(f"python rgbd_timestamp_write.py --cam_no {cam_id} --save_dir {save_dir} --max_nframes {max_nframes} --mode {mode} --node_suffix {node_suffix}")

def run_rosbag_play(rosbag_filename):
    time.sleep(3)
    os.system(f"rosbag play {rosbag_filename}")

def run_intrinsic_writer(cam_id, save_dir):
    os.system(f"python intrinsics_write.py --cam_no {cam_id} --save_dir {save_dir}")
# st()
if len(sys.argv) == 1:
  record_name = config.record_name
else:
  record_name = sys.argv[1]
record_root = os.path.join(config.data_root, record_name)
dump_folder = os.path.join(record_root, "rgb_depth_npy")
makedir(dump_folder)

dump_rgb_folder = os.path.join(dump_folder, "colorData")
makedir(dump_rgb_folder)

dump_depth_folder = os.path.join(dump_folder, "depthData")
makedir(dump_depth_folder)



# dump rgbd, timestamp to folder
bag_filename = os.path.join(record_root, "CalibData.bag")


num_cam = config.NUM_CAM


process_rosbag_play = multiprocessing.Process(target=run_rosbag_play, args=(bag_filename,))
processes = [process_rosbag_play]

for cam_id in range(1, num_cam+1):
    
    # rgb
    dump_rgb_cam_folder = os.path.join(dump_rgb_folder, f"Cam{cam_id}")
    makedir(dump_rgb_cam_folder)
    process_rgbd_writer_rgb = multiprocessing.Process(target=run_rgbd_time_writer, args=(cam_id, dump_rgb_cam_folder, "rgb",))
    processes.append(process_rgbd_writer_rgb)

    # depth
    dump_depth_cam_folder = os.path.join(dump_depth_folder, f"Cam{cam_id}")
    makedir(dump_depth_cam_folder)
    process_rgbd_writer_depth = multiprocessing.Process(target=run_rgbd_time_writer, args=(cam_id, dump_depth_cam_folder, "depth",))
    processes.append(process_rgbd_writer_depth)
    

for process in processes:
    process.start()

for process in processes:
    process.join()

# build timestap compare
# import ipdb
# ipdb.set_trace()
cmd = f"python timeCompareGeorge.py --path {dump_folder}"
# for cam_id in range(1, num_cam+1):
#    dump_rgb_cam_folder = os.path.join(dump_rgb_folder, f"Cam{cam_id}")
#    dump_depth_cam_folder = os.path.join(dump_depth_folder, f"Cam{cam_id}")

#    cmd += f" --yaml_color{cam_id}Path "
#    cmd += os.path.join(dump_rgb_cam_folder, "data.yaml")

#    cmd += f" --yaml_depth{cam_id}Path "
#    cmd += os.path.join(dump_depth_cam_folder, "data.yaml")

os.system(cmd)

cmd = f"cp  /home/zhouxian/catkin_ws/src/calibrate/src/scripts/user_scripts/smartest_calibration.npy {config.data_root}/{record_name}/ar_tag/extrinsics.npy"
print(f"written smartest_calibration to {config.data_root}/{record_name}/ar_tag")
# st()

os.system(cmd)
## write to tfrecord
#tfrecord_folder = os.path.join(record_root, "tfrecord")
#makedir(tfrecord_folder)
#os.system(f"python packtf.py --data_dir {record_root} --save_dir {tfrecord_folder}")
