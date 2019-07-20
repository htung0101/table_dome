#!/usr/bin/env python
import rospy
import argparse
import pickle
import numpy as np
from sensor_msgs.msg import CameraInfo


def callback(msg, args):   
    K = np.asarray(msg.K).reshape(3, 3)
    args[0].append(K)


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True,\
        help='specify the directory you want to save data to')
    parser.add_argument('--cam_no', type=str, required=True,\
        help='for which camera')
    args=parser.parse_args()


    #with open(kwargs['pkl_file_path'], 'rb') as f:
    #    save_dict = pickle.load(f)
    #f.close()

    rospy.init_node('info_subscribe')

    list_intrinsics = []
    sub_name = '/camera{}/aligned_depth_to_color/camera_info'.format(args.cam_no)
    data = rospy.wait_for_message(sub_name, CameraInfo,)

    callback(data, (list_intrinsics))
    

    with open(os.path.join(args.save_dir, "intrinsics.npy"),'wb') as f:
        np.save(f, list_intrinsics[0])
    print("intrinsics", list_intrinsics[0])


if __name__ == '__main__':
     main(pkl_file_path="/home/zhouxian/catkin_ws/src/calibrate/src/scripts/all_data.pkl")
