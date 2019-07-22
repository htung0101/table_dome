#!/usr/bin/env python
import rospy
import pickle
import numpy as np
from sensor_msgs.msg import CameraInfo

camera_1_rec_flag = False
camera_2_rec_flag = False
camera_3_rec_flag = False
camera_4_rec_flag = False
camera_5_rec_flag = False
camera_6_rec_flag = False

def infoCallback1(msg, args):
    global camera_1_rec_flag
    if not camera_1_rec_flag:
        print('got camera 1')
        cam_no = args[0]
        save_dict = args[1]
        K = np.asarray(msg.K).reshape(3, 3)
        save_dict['cam_{}_intrinsics'.format(cam_no)] = K
        camera_1_rec_flag = True

def infoCallback2(msg, args):
    global camera_2_rec_flag
    if not camera_2_rec_flag:
        print('got camera 2')
        cam_no = args[0]
        save_dict = args[1]
        K = np.asarray(msg.K).reshape(3, 3)
        save_dict['cam_{}_intrinsics'.format(cam_no)] = K
        camera_2_rec_flag = True

def infoCallback3(msg, args):
    global camera_3_rec_flag
    if not camera_3_rec_flag:
        print('got camera 3')
        cam_no = args[0]
        save_dict = args[1]
        K = np.asarray(msg.K).reshape(3, 3)
        save_dict['cam_{}_intrinsics'.format(cam_no)] = K
        camera_3_rec_flag = True

def infoCallback4(msg, args):
    global camera_4_rec_flag
    if not camera_4_rec_flag:
        print('got camera 4')
        cam_no = args[0]
        save_dict = args[1]
        K = np.asarray(msg.K).reshape(3, 3)
        save_dict['cam_{}_intrinsics'.format(cam_no)] = K
        camera_4_rec_flag = True

def infoCallback5(msg, args):
    global camera_5_rec_flag
    if not camera_5_rec_flag:
        print('got camera 5')
        cam_no = args[0]
        save_dict = args[1]
        K = np.asarray(msg.K).reshape(3, 3)
        save_dict['cam_{}_intrinsics'.format(cam_no)] = K
        camera_5_rec_flag = True

def infoCallback6(msg, args):
    global camera_6_rec_flag
    if not camera_6_rec_flag:
        print('got camera 6')
        cam_no = args[0]
        save_dict = args[1]
        K = np.asarray(msg.K).reshape(3, 3)
        save_dict['cam_{}_intrinsics'.format(cam_no)] = K
        camera_6_rec_flag = True

def main(**kwargs):
    with open(kwargs['pkl_file_path'], 'rb') as f:
        save_dict = pickle.load(f)
    f.close()

    rospy.init_node('info_subscribe')
    rospy.Subscriber('/camera1/aligned_depth_to_color/camera_info',\
        CameraInfo, infoCallback1, (1, save_dict))
    rospy.Subscriber('/camera2/aligned_depth_to_color/camera_info',\
        CameraInfo, infoCallback2, (2, save_dict))
    rospy.Subscriber('/camera3/aligned_depth_to_color/camera_info',\
        CameraInfo, infoCallback3, (3, save_dict))
    rospy.Subscriber('/camera4/aligned_depth_to_color/camera_info',\
        CameraInfo, infoCallback4, (4, save_dict))
    rospy.Subscriber('/camera5/aligned_depth_to_color/camera_info',\
        CameraInfo, infoCallback5, (5, save_dict))
    rospy.Subscriber('/camera6/aligned_depth_to_color/camera_info',\
        CameraInfo, infoCallback6, (6, save_dict))

    import pdb; pdb.set_trace()

    rospy.spin()

if __name__ == '__main__':
     main(pkl_file_path="/home/zhouxian/catkin_ws/src/calibrate/src/scripts/all_data_new_calib.pkl")
