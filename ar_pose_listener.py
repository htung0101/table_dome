#!/usr/bin/env python

import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
import pdb
import pickle
import os
import argparse
import numpy as np

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

    pose_dict = {
        'position': np.asarray([x, y, z]),
        'orientation': np.asarray([qx, qy, qz, qw])
    }
    
    with open('/home/zhouxian/catkin_ws/src/calibrate/src/scripts/cam_{}_pose1111111111.txt'.format(cam_no), 'wb') as f:
        pickle.dump(pose_dict, f)

    f.close()
    print('done you can close me now')


def listener():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_no', type=int, required=True)
    args = parser.parse_args()

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('ar_pose_marker', AlvarMarkers, callback, (args.cam_no,))


    rospy.spin()

if __name__ == '__main__':
    listener()