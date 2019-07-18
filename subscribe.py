#!/usr/bin/env python
from message_filters import ApproximateTimeSynchronizer, Subscriber
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import rosbag

'''

def gotimage(image1,image2):
    print "CALLBACK"
    print "Image1: ",image1.header.stamp
    print "Image2: ",image2.header.stamp
    #print "got Synced Images"
    pub1 = rospy.Publisher('/camera1/color/image_raw_throttle_sync',Image, queue_size=10)
    pub1.publish(image1)

    pub2 = rospy.Publisher('/camera2/color/image_raw_throttle_sync',Image, queue_size=10)
    pub2.publish(image2)


rospy.init_node('subscribe')
image1_sub = Subscriber('/camera1/color/image_raw_throttle', Image)
image2_sub = Subscriber('/camera2/color/image_raw_throttle', Image)

ats = ApproximateTimeSynchronizer([image1_sub, image2_sub], queue_size=10, slop=0.2)
ats.registerCallback(gotimage)
rospy.spin()

'''
def gotimage(image1,image2,image3,image4,image5,image6):
    print "CALLBACK"
    print "Image1: ",image1.header.stamp
    print "Image2: ",image2.header.stamp
    print "Image3: ",image3.header.stamp
    print "Image4: ",image4.header.stamp
    print "Image5: ",image5.header.stamp
    print "Image6: ",image6.header.stamp
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


rospy.init_node('subscribe')
image1_sub = Subscriber('/camera1/color/image_raw_throttle', Image)
image2_sub = Subscriber('/camera2/color/image_raw_throttle', Image)
image3_sub = Subscriber('/camera3/color/image_raw_throttle', Image)
image4_sub = Subscriber('/camera4/color/image_raw_throttle', Image)
image5_sub = Subscriber('/camera5/color/image_raw_throttle', Image)
image6_sub = Subscriber('/camera6/color/image_raw_throttle', Image)

ats = ApproximateTimeSynchronizer([image1_sub, image2_sub, image3_sub, image4_sub, image5_sub, image6_sub], queue_size=10, slop=0.2)
ats.registerCallback(gotimage)
rospy.spin()

'''
rospy.init_node('subscribe')
bag=rosbag.Bag('calibCam.bag','w')
image_sub = rospy.Subscriber("/camera3/color/image_raw",Image,gotimage1)
rospy.spin()
'''