#!/usr/bin/env python
from message_filters import ApproximateTimeSynchronizer, Subscriber, TimeSynchronizer
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import rosbag

def pubLishImages(image1, image2, image3, image4, image5, image6, adepth1, adepth2, adepth3, adepth4, adepth5, adepth6):
    """
    print("CALLBACK")
    print('---------- TIME ------------')
    print("Image1: ",image1.header.stamp)
    print("Image2: ",image2.header.stamp)
    print("Image3: ",image3.header.stamp)
    print("Image4: ",image4.header.stamp)
    print("Image5: ",image5.header.stamp)
    print("Image6: ",image6.header.stamp)
    print("adepth1: ",adepth1.header.stamp)
    print("adepth2: ",adepth2.header.stamp)
    print("adepth3: ",adepth3.header.stamp)
    print("adepth4: ",adepth4.header.stamp)
    print("adepth5: ",adepth5.header.stamp)
    print("adepth6: ",adepth6.header.stamp)
    """
    # NOTE: save all these in the directory
   
    #print "got Synced Images"
    pub1 = rospy.Publisher('/camera1/color/image_raw_throttle_sync',Image, queue_size=10)
    pub1.publish(image1)

    pub2 = rospy.Publisher('/camera2/color/image_raw_throttle_sync',Image, queue_size=10)
    pub2.publish(image2)

    pub3 = rospy.Publisher('/camera3/color/image_raw_throttle_sync',Image, queue_size=10)
    pub3.publish(image3)

    pub4 = rospy.Publisher('/camera4/color/image_raw_throttle_sync',Image, queue_size=10)
    pub4.publish(image4)

    pub5 = rospy.Publisher('/camera5/color/image_raw_throttle_sync',Image, queue_size=10)
    pub5.publish(image5)

    pub6 = rospy.Publisher('/camera6/color/image_raw_throttle_sync',Image, queue_size=10)
    pub6.publish(image6)

    pub7= rospy.Publisher('/camera1/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub7.publish(adepth1)

    pub8 = rospy.Publisher('/camera2/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub8.publish(adepth2)

    pub9 = rospy.Publisher('/camera3/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub9.publish(adepth3)

    pub10 = rospy.Publisher('/camera4/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub10.publish(adepth4)

    pub11 = rospy.Publisher('/camera5/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub11.publish(adepth5)

    pub12 = rospy.Publisher('/camera6/aligned_depth_to_color/image_raw_throttle_sync',Image, queue_size=10)
    pub12.publish(adepth6)



rospy.init_node('subscribe')
image1_sub = Subscriber('/camera1/color/image_raw_throttle', Image)
image2_sub = Subscriber('/camera2/color/image_raw_throttle', Image)
image3_sub = Subscriber('/camera3/color/image_raw_throttle', Image)
image4_sub = Subscriber('/camera4/color/image_raw_throttle', Image)
image5_sub = Subscriber('/camera5/color/image_raw_throttle', Image)
image6_sub = Subscriber('/camera6/color/image_raw_throttle', Image)

aligned_depth1_sub=Subscriber('/camera1/aligned_depth_to_color/image_raw_throttle', Image)
aligned_depth2_sub=Subscriber('/camera2/aligned_depth_to_color/image_raw_throttle', Image)
aligned_depth3_sub=Subscriber('/camera3/aligned_depth_to_color/image_raw_throttle', Image)
aligned_depth4_sub=Subscriber('/camera4/aligned_depth_to_color/image_raw_throttle', Image)
aligned_depth5_sub=Subscriber('/camera5/aligned_depth_to_color/image_raw_throttle', Image)
aligned_depth6_sub=Subscriber('/camera6/aligned_depth_to_color/image_raw_throttle', Image)

ats = ApproximateTimeSynchronizer([image1_sub, image2_sub, image3_sub, image4_sub, image5_sub, image6_sub, aligned_depth1_sub, aligned_depth2_sub, aligned_depth3_sub, aligned_depth4_sub, aligned_depth5_sub, aligned_depth6_sub], queue_size=30, slop=0.30)

ats.registerCallback(pubLishImages)

try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
