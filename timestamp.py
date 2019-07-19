#!usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import argparse
import yaml

cnt=0

def callback(image, args):
    global cnt
    #print(image.header.seq)
    #print(image.header.stamp)
    data={cnt:{'Timestamp': image.header.stamp}}
    cnt+=1
    args[0].append(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_no', type=str, required=True,\
        help='for which camera')
    parser.add_argument('--save_dir', type=str, required=True,\
        help='directory to save the timestamp yaml')
    parser.add_argument('--mode', type=str, required=True,\
        help='[depth, rgb]')
    parser.add_argument('--max_nframes', type=int, default=10000, required=False,\
        help='for which camera')

    args = parser.parse_args()
    rospy.init_node('TimeStampPrint')

    if args.mode == "rgb":
        sub_name = '/camera{}/color/image_raw_throttle_sync'.format(args.cam_no)
    elif args.mode == "depth":
        sub_name = '/camera{}/aligned_depth_to_color/image_raw_throttle_sync'.format(args.cam_no)
    else:
        raise Exception("no such mode for timestamp:", args.mode)

    list_msg = []
    for i in range(args.max_nframes):
        try:
            data = rospy.wait_for_message(sub_name, Image, timeout=3.0 if i>0 else None)
        except rospy.exceptions.ROSException:
            total_images = i
            print("Total number of {} images: ".format(args.mode), i)
            break
        callback(data, (list_msg,))

    output_filename = os.path.join(args.save_dir, "data.yaml")
    print("yaml_output_filename", output_filename)

    with open(output_filename, 'w') as outfile:
        for data in list_msg:
            yaml.dump(data, outfile, default_flow_style=False)
