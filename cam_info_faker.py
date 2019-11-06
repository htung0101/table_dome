import rospy
from sensor_msgs.msg import CameraInfo
import yaml


def yaml_to_CameraInfo(yaml_fname, cam_name):
    with open(yaml_fname, "r") as file_handle:
        calib_data = yaml.load(file_handle)

    all_cam_info = calib_data[cam_name]
    camera_info_msg = CameraInfo()
    camera_info_msg.width = all_cam_info['resolution'][0]
    camera_info_msg.height = all_cam_info['resolution'][1]

    cam_intrinsics = all_cam_info['intrinsics']
    camera_info_msg.K = [cam_intrinsics[0], 0.0, cam_intrinsics[2], 0.0, cam_intrinsics[1], cam_intrinsics[3], 0.0, 0.0, 1.0]
    camera_info_msg.D = [0.0, 0.0, 0.0, 0.0]
    camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    camera_info_msg.P = [cam_intrinsics[0], 0.0, cam_intrinsics[2], 0.0, 0.0, cam_intrinsics[1], cam_intrinsics[3], 0.0, 0.0, 0.0, 1.0, 0.0]
    camera_info_msg.distortion_model = "plumb_bob"
    return camera_info_msg


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("filename", help="Path to yaml file containing " +\
                                             "camera calibration data")
    args = arg_parser.parse_args()
    filename = args.filename

    # Parse yaml file
    cam_names = ['cam2', 'cam3']
    cam_infos = list()
    for i in cam_names:
        cam_infos.append(yaml_to_CameraInfo(filename, i))

    # Initialize publisher node
    rospy.init_node("camera_info_publisher", anonymous=True)
    publisher3 = rospy.Publisher("camera_info3", CameraInfo, queue_size=10)
    publisher4 = rospy.Publisher("camera_info4", CameraInfo, queue_size=10)
    rate = rospy.Rate(10)

    # Run publisher
    while not rospy.is_shutdown():
        publisher3.publish(cam_infos[0])
        publisher4.publish(cam_infos[1])
        rate.sleep()