#!usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import argparse
import pickle

# TODO: Remove this global declaration
bridge = CvBridge()
cnt = 0
cnt1 = 0

def callback1(depth, args):
    print('coming here')
    global cnt
    global cnt1
    save_dir = args[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print('got a depth image')
    cv_image = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
    '''
    save_path = os.path.join(save_dir, 'color_img_{}_{}.jpg'.format(args[1], cnt))
    cv2.imwrite(save_path, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
    cnt+=1
    print(cv_image)
    '''
    with open(os.path.join(save_dir, "cam_{}_{}_color.npy".format(args[1],cnt1)),'wb') as f:
        #pickle.dump(cv_image, f)
        np.save(f,cv_image)
        cnt1+=1
    f.close()
    '''
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_image = bridge.imgmsg_to_cv2(depth, "passthrough")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        save_path = os.path.join(save_dir, 'color_img_{}_{}.jpg'.format(args[1], cnt))
        #import pdb; pdb.set_trace()
        #cv2.imwrite(save_path,cv2_img)
        cv2.imwrite(save_path, cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR))
        cnt += 1
    '''

    print('done')



    print('done')
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True,\
        help='specify the directory you want to save data to')
    parser.add_argument('--cam_no', type=str, required=True,\
        help='for which camera')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        print("Path doesnt exist!")
        os.makedirs(args.save_dir)

    rospy.init_node('colorSubscribe')
    # depth_sub=rospy.Subscriber('/{}/aligned_depth_to_color/image_raw_throttle_sync'.format(args.cam_no),\
    #     Image,callback1, (args.save_dir, args.cam_no))
    depth_sub=rospy.Subscriber('/{}/color/image_raw_throttle_sync'.format(args.cam_no),\
        Image,callback1, (args.save_dir, args.cam_no))
    rospy.sleep(2)
    rospy.spin()
    '''
    rosbag record -O newCalibData1.bag /camera1/color/image_raw_throttle_sync /camera1/aligned_depth_to_color/image_raw_throttle_sync /camera2/color/image_raw_throttle_sync /camera2/aligned_depth_to_color/image_raw_throttle_sync /camera3/color/image_raw_throttle_sync /camera3/aligned_depth_to_color/image_raw_throttle_sync /camera4/color/image_raw_throttle_sync /camera4/aligned_depth_to_color/image_raw_throttle_sync /camera5/color/image_raw_throttle_sync /camera5/aligned_depth_to_color/image_raw_throttle_sync /camera6/color/image_raw_throttle_sync /camera6/aligned_depth_to_color/image_raw_throttle_sync

    rosbag record -O newCalib1.bag /camera1/color/image_raw_throttle_sync /camera2/color/image_raw_throttle_sync /camera3/color/image_raw_throttle_sync /camera4/color/image_raw_throttle_sync /camera5/color/image_raw_throttle_sync /camera6/color/image_raw_throttle_sync
    '''
