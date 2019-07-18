#!/usr/bin/env python
from message_filters import TimeSynchronizer, Subscriber
import rospy
from sensor_msgs.msg import Image

def pubLishImages1(image1,adepth1):
    # print "CALLBACK"
    print('---------- TIME ------------')
    print("Image1: ",image1.header.stamp)
    print("Depth1: ",adepth1.header.stamp)

    #print "got Synced Images"
    pub= rospy.Publisher('/camera1/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub.publish(adepth1)

def pubLishImages2(image2,adepth2):
    # print "CALLBACK"
    print('---------- TIME ------------')
    print("Image2: ",image2.header.stamp)
    print("Depth2: ",adepth2.header.stamp)

    #print "got Synced Images"
    pub= rospy.Publisher('/camera2/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub.publish(adepth2)

def pubLishImages3(image3,adepth3):
    # print "CALLBACK"
    print('---------- TIME ------------')
    print("Image3: ",image3.header.stamp)
    print("Depth3: ",adepth3.header.stamp)

    #print "got Synced Images"
    pub= rospy.Publisher('/camera3/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub.publish(adepth3)

def pubLishImages4(image4,adepth4):
    # print "CALLBACK"
    print('---------- TIME ------------')
    print("Image4: ",image4.header.stamp)
    print("Depth4: ",adepth4.header.stamp)

    #print "got Synced Images"
    pub= rospy.Publisher('/camera4/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub.publish(adepth4)

def pubLishImages5(image5,adepth5):
    # print "CALLBACK"
    print('---------- TIME ------------')
    print("Image5: ",image5.header.stamp)
    print("Depth5: ",adepth5.header.stamp)

    #print "got Synced Images"
    pub= rospy.Publisher('/camera5/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub.publish(adepth5)

def pubLishImages6(image6,adepth6):
    # print "CALLBACK"
    print('---------- TIME ------------')
    print("Image6: ",image6.header.stamp)
    print("Depth6: ",adepth6.header.stamp)

    #print "got Synced Images"
    pub= rospy.Publisher('/camera6/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub.publish(adepth6)


rospy.init_node('subscribe2')
image1_sub = Subscriber('/camera1/color/image_raw_throttle_sync', Image)
image2_sub = Subscriber('/camera2/color/image_raw_throttle_sync', Image)
image3_sub = Subscriber('/camera3/color/image_raw_throttle_sync', Image)
image4_sub = Subscriber('/camera4/color/image_raw_throttle_sync', Image)
image5_sub = Subscriber('/camera5/color/image_raw_throttle_sync', Image)
image6_sub = Subscriber('/camera6/color/image_raw_throttle_sync', Image)

aligned_depth1_sub=Subscriber('/camera1/aligned_depth_to_color/image_raw', Image)
aligned_depth2_sub=Subscriber('/camera2/aligned_depth_to_color/image_raw', Image)
aligned_depth3_sub=Subscriber('/camera3/aligned_depth_to_color/image_raw', Image)
aligned_depth4_sub=Subscriber('/camera4/aligned_depth_to_color/image_raw', Image)
aligned_depth5_sub=Subscriber('/camera5/aligned_depth_to_color/image_raw', Image)
aligned_depth6_sub=Subscriber('/camera6/aligned_depth_to_color/image_raw', Image)


ts1 = TimeSynchronizer([image1_sub, aligned_depth1_sub],10)
ts2 = TimeSynchronizer([image2_sub, aligned_depth2_sub],10)
ts3 = TimeSynchronizer([image3_sub, aligned_depth3_sub],10)
ts4 = TimeSynchronizer([image4_sub, aligned_depth4_sub],10)
ts5 = TimeSynchronizer([image5_sub, aligned_depth5_sub],10)
ts6 = TimeSynchronizer([image6_sub, aligned_depth6_sub],10)

ts1.registerCallback(pubLishImages1)
ts2.registerCallback(pubLishImages2)
ts3.registerCallback(pubLishImages3)
ts4.registerCallback(pubLishImages4)
ts5.registerCallback(pubLishImages5)
ts6.registerCallback(pubLishImages6)

rospy.spin()







