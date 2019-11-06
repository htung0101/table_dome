import numpy as np
import rospy
import copy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
def callback3(image):
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
    global image3
    image3 = copy.deepcopy(cv_image)


def callback4(image):
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
    global image4
    image4 = copy.deepcopy(cv_image)

def callback_color3(image):
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
    global color_image3
    color_image3 = copy.deepcopy(cv_image)


def callback_color4(image):
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
    global color_image4
    color_image4 = copy.deepcopy(cv_image)
    

if __name__ == '__main__':
	rospy.init_node('depthSubscribe')

	sub_name3 = '/camera{}/aligned_depth_to_color/image_raw'.format(3)
	sub_name4 = '/camera{}/aligned_depth_to_color/image_raw'.format(4)

	sub_color_name3 = '/camera3/color/image_raw'
	sub_color_name4 = '/camera4/color/image_raw'

	rospy.Subscriber(sub_name3, Image, callback3)
	rospy.Subscriber(sub_name4, Image, callback4)

	rospy.Subscriber(sub_color_name3, Image, callback_color3)
	rospy.Subscriber(sub_color_name4, Image, callback_color4)

	rospy.sleep(2)

	depth_cam3 = copy.deepcopy(image3)
	depth_cam4 = copy.deepcopy(image4)

	color_cam3 = copy.deepcopy(color_image3)
	color_cam4 = copy.deepcopy(color_image4)

	depths_1105 = np.stack((depth_cam3, depth_cam4))
	np.save('depths_1105.npy', depths_1105)

	color_1105 = np.stack((color_cam3, color_cam4))
	np.save('colors_1105.npy', color_1105)