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

def callback(image):
    global cnt
    print(image.header.seq)
    print(image.header.stamp)
    data={cnt:{'Timestamp': image.header.stamp}}
    cnt+=1

    with open('data.yaml', 'a+') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    outfile.close()


if __name__ == "__main__":
    rospy.init_node('TimeStampPrint')
    image_sub = rospy.Subscriber('/camera6/color/image_raw_throttle_sync',Image,callback)
    #import pdb;pdb.set_trace()
    rospy.spin()