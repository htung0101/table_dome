#!/usr/bin/env python

import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
import pdb
import pickle
import os
import argparse
import numpy as np
import ipdb
st = ipdb.set_trace

def callback(data, args):
    cam_no = args[0]
    x = data.markers[0].pose.pose.position.x
    y = data.markers[0].pose.pose.position.y
    z = data.markers[0].pose.pose.position.z

    # get the orientation
    qx = data.markers[0].pose.pose.orientation.x
    qy = data.markers[0].pose.pose.orientation.y
    qz = data.markers[0].pose.pose.orientation.z
    qw = data.markers[0].pose.pose.orientation.w
    print("writing ar infor in to", args[1])
    pose_dict = {
        'position': np.asarray([x, y, z]),
        'orientation': np.asarray([qx, qy, qz, qw])
    }
    
    #'/home/zhouxian/catkin_ws/src/calibrate/src/scripts/cam_{}_pose1111111111.txt'.format(cam_no)

    with open(args[1], 'wb') as f:
        pickle.dump(pose_dict, f)

    f.close()


def listener():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_no', type=int, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('ar_pose_marker', AlvarMarkers, callback, (args.cam_no, args.out))
    data = rospy.wait_for_message("ar_pose_marker", AlvarMarkers)
    callback(data, (args.cam_no, args.out))


if __name__ == '__main__':
    listener()