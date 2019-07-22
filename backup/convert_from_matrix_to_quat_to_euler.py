import numpy as np
import tf
import rospy
import pickle

def main():
    with open('/home/zhouxian/all_data_with_intrinsics.pkl', 'rb') as f:
        data = pickle.load(f)
    f.close()

    print(data.keys())

    tf_0_to_1 = data['cam_0_to_1']
    # now convert this to euler notation and position, but first to quaternion
    pos1 = tf_0_to_1[:3, 3]
    tf_1_quat = tf.transformations.quaternion_from_matrix(tf_0_to_1)
    tf_1_rpy = tf.transformations.euler_from_quaternion(tf_1_quat)

    # for camera 2
    tf_0_to_2 = data['cam_0_to_2']
    pos2 = tf_0_to_2[:3, 3]
    tf_2_quat = tf.transformations.quaternion_from_matrix(tf_0_to_2)
    tf_2_rpy = tf.transformations.euler_from_quaternion(tf_2_quat)

    # for camera 3
    tf_0_to_3 = data['cam_0_to_3']
    pos3 = tf_0_to_3[:3, 3]
    tf_3_quat = tf.transformations.quaternion_from_matrix(tf_0_to_3)
    tf_3_rpy = tf.transformations.euler_from_quaternion(tf_3_quat)

    # for camera 4
    tf_0_to_4 = data['cam_0_to_4']
    pos4 = tf_0_to_4[:3, 3]
    tf_4_quat = tf.transformations.quaternion_from_matrix(tf_0_to_4)
    tf_4_rpy = tf.transformations.euler_from_quaternion(tf_4_quat)

    # for camera 5
    tf_0_to_5 = data['cam_0_to_5']
    pos5 = tf_0_to_5[:3, 3]
    tf_5_quat = tf.transformations.quaternion_from_matrix(tf_0_to_5)
    tf_5_rpy = tf.transformations.euler_from_quaternion(tf_5_quat)

    print(pos1)
    print(pos2)
    print(pos3)
    print(pos4)
    print(pos5)

    from IPython import embed; embed()

    print(tf_1_rpy)
    print(tf_2_rpy)
    print(tf_3_rpy)
    print(tf_4_rpy)
    print(tf_5_rpy)

    from IPython import embed; embed()

if __name__ == '__main__':
    main()