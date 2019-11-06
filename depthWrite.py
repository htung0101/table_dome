#!usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import argparse
import pickle
#import ipdb
#st=ipdb.set_trace
# TODO: Remove this global declaration
bridge = CvBridge()
cnt = 0
cnt1 = 0

def callback(image, args):
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
    args[0].append(cv_image)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True,\
        help='specify the directory you want to save data to')
    parser.add_argument('--cam_no', type=str, required=True,\
        help='for which camera')
    parser.add_argument('--mode', type=str, required=True,\
        help='[depth, rgb]')
    parser.add_argument('--max_nframes', type=int, default=10000, required=False,\
        help='for which camera')

    args = parser.parse_args()

    #if not os.path.exists(args.save_dir):
    #    print("Path doesnt exist!")
    #    os.makedirs(args.save_dir)

    rospy.init_node('colorSubscribe')


    if args.mode == "depth":
        sub_name = '/camera{}/aligned_depth_to_color/image_raw_throttle_sync'.format(args.cam_no)
        prefix = "depth"
    elif args.mode == "rgb":
        sub_name = '/camera{}/color/image_raw_throttle_sync'.format(args.cam_no)
        prefix = "color"
    else:
        raise Exception("no such mode for depthWrite")

    images_list = []
    for i in range(args.max_nframes):
        try:
            data = rospy.wait_for_message(sub_name, Image, timeout=3.0 if i>0 else None)
        except rospy.exceptions.ROSException:
            print("Total number of {} images: ".format(args.mode), i)
            break
        callback(data, (images_list,))
        
    print("number of {}:".format(args.mode), len(images_list))
    all_images = np.stack(images_list, 0)
    with open(os.path.join(args.save_dir, "cam_{}_{}.npy".format(args.cam_no, prefix)),'wb') as f:
        np.save(f, all_images)
