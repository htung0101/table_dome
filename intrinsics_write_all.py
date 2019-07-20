#!/usr/bin/env python
import rospy
import argparse
import pickle
import numpy as np
from sensor_msgs.msg import CameraInfo
import os

def callback(msg, args):   
    K = np.asarray(msg.K).reshape(3, 3)
    cam_id = args[0]
    args[1][str(cam_id)] = K


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True,\
        help='specify the directory you want to save data to')
    parser.add_argument('--num_cam', type=int, required=True,\
        help='for which camera')
    args=parser.parse_args()


    #with open(kwargs['pkl_file_path'], 'rb') as f:
    #    save_dict = pickle.load(f)
    #f.close()

    rospy.init_node('info_subscribe')

    list_intrinsics = dict()
    for cam_id in range(1, args.num_cam+1):
        sub_name = '/camera{}/aligned_depth_to_color/camera_info'.format(cam_id)
        data = rospy.wait_for_message(sub_name, CameraInfo,)
        callback(data, (cam_id, list_intrinsics))
    

    with open(os.path.join(args.save_dir, "intrinsics.pkl"),'wb') as f:
        pickle.dump(list_intrinsics, f)


if __name__ == '__main__':
     main(pkl_file_path="/home/zhouxian/catkin_ws/src/calibrate/src/scripts/all_data.pkl")
