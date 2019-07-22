#!usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import argparse

# TODO: Remove this global declaration
bridge = CvBridge()
cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
cnt5 = 0
cnt6 = 0

def cam6_callback(img, args):
    global cnt6
    save_dir = args[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('got a color image')
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img, "passthrough")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        save_path = os.path.join(save_dir, 'color_img{}_{}.jpg'.format(args[1], cnt6))
        cv2.imwrite(save_path, cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR))
        cnt6 += 1

def cam5_callback(img, args):
    global cnt5
    save_dir = args[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('got a color image')
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img, "passthrough")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        save_path = os.path.join(save_dir, 'color_img{}_{}.jpg'.format(args[1], cnt5))

        cv2.imwrite(save_path, cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR))
        cnt5 += 1

def cam4_callback(img, args):
    global cnt4
    save_dir = args[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('got a color image')
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img, "passthrough")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        save_path = os.path.join(save_dir, 'color_img{}_{}.jpg'.format(args[1], cnt4))

        cv2.imwrite(save_path, cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR))
        cnt4 += 1

def cam3_callback(img, args):
    global cnt3
    save_dir = args[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('got a color image')
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img, "passthrough")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        save_path = os.path.join(save_dir, 'color_img{}_{}.jpg'.format(args[1], cnt3))

        cv2.imwrite(save_path, cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR))
        cnt3 += 1

def cam2_callback(img, args):
    global cnt2
    save_dir = args[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('got a color image')
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img, "passthrough")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        save_path = os.path.join(save_dir, 'color_img{}_{}.jpg'.format(args[1], cnt2))

        cv2.imwrite(save_path, cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR))
        cnt2 += 1

def cam1_callback(img, args):
    global cnt1
    save_dir = args[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('got a color image')
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img, "passthrough")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        save_path = os.path.join(save_dir, 'color_img{}_{}.jpg'.format(args[1], cnt1))

        cv2.imwrite(save_path, cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR))
        cnt1 += 1
    
if __name__=="__main__":
    rospy.init_node('colorSubscribe')

    rospy.sleep(10)

    color1_sub=rospy.Subscriber('/camera1/color/image_raw',\
        Image, cam1_callback, ("/home/zhouxian/empty_table/1", 1))

    color2_sub=rospy.Subscriber('/camera2/color/image_raw',\
        Image, cam2_callback, ("/home/zhouxian/empty_table/2", 2))
    
    color3_sub=rospy.Subscriber('/camera3/color/image_raw',\
        Image, cam3_callback, ("/home/zhouxian/empty_table/3", 3))
    
    color4_sub=rospy.Subscriber('/camera4/color/image_raw',\
        Image, cam4_callback, ("/home/zhouxian/empty_table/4", 4))
    
    color5_sub=rospy.Subscriber('/camera5/color/image_raw',\
        Image, cam5_callback, ("/home/zhouxian/empty_table/5", 5))
    
    color6_sub=rospy.Subscriber('/camera6/color/image_raw',\
        Image, cam6_callback, ("/home/zhouxian/empty_table/6", 6))
    
    
    rospy.sleep(2)
    rospy.spin()

    '''
    kalibr_calibrate_cameras --target april_6x6.yaml --bag newCalib.bag --models pinhole-equi pinhole-equi pinhole-equi pinhole-equi pinhole-equi pinhole-equi --topics /camera1/color/image_raw /camera2/color/image_raw /camera3/color/image_raw /camera4/color/image_raw /camera5/color/image_raw /camera6/color/image_raw
    '''
