import rospy
import tf
from tf import TransformListener
import numpy as np

def main():
    c1Link_to_c1ColorOpticalFrame_pos = np.asarray([-0.000, 0.015, 0.000])
    c1Link_to_c1ColorOpticalFrame_rot = tf.transformations.quaternion_matrix(np.asarray([0.506, -0.494, 0.507, -0.492]))

    mat = np.asarray([[ 0.4060203 ,  0.75756728, -0.51111576,  0.17559398],
       [-0.76612722,  0.58706488,  0.26154141, -0.0421986 ],
       [ 0.49819333,  0.28538857,  0.81875318,  0.0133699 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    c2ColorOpticalFrame_to_c2Link_pos = np.asarray([0.015, 0.001, -0.000])
    c2ColorOpticalFrame_to_c2Link_rot = tf.transformations.quaternion_matrix(np.asarray([0.507, -0.495, 0.501, 0.497]))

    # now form the transformation matrices
    c1Link_to_c1ColorOpticalFrame = c1Link_to_c1ColorOpticalFrame_rot
    c1Link_to_c1ColorOpticalFrame[:3, 3] = c1Link_to_c1ColorOpticalFrame_pos

    c2ColorOpticalFrame_to_c2Link = c2ColorOpticalFrame_to_c2Link_rot
    c2ColorOpticalFrame_to_c2Link[:3, 3] = c2ColorOpticalFrame_to_c2Link_pos

    c1Link_to_c2ColorOpticalFrame = np.dot(c1Link_to_c1ColorOpticalFrame, tf.transformations.inverse_matrix(mat))
    c1Link_to_c2Link = np.dot(c1Link_to_c2ColorOpticalFrame, c2ColorOpticalFrame_to_c2Link)

    c1Link_to_c2Link_pos = c1Link_to_c2Link[:3, 3]
    c1Link_to_c2Link_quat = tf.transformations.quaternion_from_matrix(c1Link_to_c2Link)

    # .................. for camera3 to camera1 .................................... #
    
    c1_to_c3_mat = np.asarray([[-0.32151113,  0.78765464, -0.5255766 ,  0.34445858],
       [-0.70964797,  0.1670522 ,  0.68446572, -0.18581281],
       [ 0.62692133,  0.59303771,  0.50524837,  0.13664849],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    c3ColorOpticalFrame_to_c3Link_pos = np.asarray([0.015, 0.001, 0.000])
    c3ColorOpticalFrame_to_c3Link_rot = tf.transformations.quaternion_matrix(np.asarray([0.505, -0.495, 0.504, 0.495]))
    c3ColorOpticalFrame_to_c3Link = c3ColorOpticalFrame_to_c3Link_rot
    c3ColorOpticalFrame_to_c3Link[:3, 3] = c3ColorOpticalFrame_to_c3Link_pos

    c1Link_to_c3ColorOpticalFrame = np.dot(c1Link_to_c1ColorOpticalFrame, tf.transformations.inverse_matrix(c1_to_c3_mat))
    c1Link_to_c3Link = np.dot(c1Link_to_c3ColorOpticalFrame, c3ColorOpticalFrame_to_c3Link)

    c1Link_to_c3Link_pos = c1Link_to_c3Link[:3, 3]
    c1Link_to_c3Link_quat = tf.transformations.quaternion_from_matrix(c1Link_to_c3Link)

    # .................. for camera4 to camera1 ..................................... #
    c1_to_c4_mat = np.asarray([[-0.98034758, -0.19519336, -0.02860359, -0.01197254],
       [-0.00334246, -0.12853606,  0.9916992 , -0.37062505],
       [-0.19724969,  0.97230552,  0.12535759,  0.34614991],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
    c4ColorOpticalFrame_to_c4Link_pos = np.asarray([0.015, 0.000, 0.000])
    c4ColorOpticalFrame_to_c4Link_rot = tf.transformations.quaternion_matrix(np.asarray([0.506, -0.495, 0.504, 0.495]))
    c4ColorOpticalFrame_to_c4Link = c4ColorOpticalFrame_to_c4Link_rot
    c4ColorOpticalFrame_to_c4Link[:3, 3] = c4ColorOpticalFrame_to_c4Link_pos

    c1Link_to_c4ColorOpticalFrame = np.dot(c1Link_to_c1ColorOpticalFrame, tf.transformations.inverse_matrix(c1_to_c4_mat))
    c1Link_to_c4Link = np.dot(c1Link_to_c4ColorOpticalFrame, c4ColorOpticalFrame_to_c4Link)

    c1Link_to_c4Link_pos = c1Link_to_c4Link[:3, 3]
    c1Link_to_c4Link_quat = tf.transformations.quaternion_from_matrix(c1Link_to_c4Link)

    # .................. for camera5 to camera1 ..................................... #
    c1_to_c5_mat = np.asarray([[-0.48201458, -0.77182315,  0.41466972, -0.20204585],
       [ 0.63925651,  0.0138576 ,  0.7688687 , -0.31471432],
       [-0.59917698,  0.63568624,  0.48671341,  0.18944575],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    c5ColorOpticalFrame_to_c5Link_pos = np.asarray([0.015, 0.000, 0.000])
    c5ColorOpticalFrame_to_c5Link_rot = tf.transformations.quaternion_matrix(np.asarray([0.505, -0.496, 0.500, 0.499]))
    c5ColorOpticalFrame_to_c5Link = c5ColorOpticalFrame_to_c5Link_rot
    c5ColorOpticalFrame_to_c5Link[:3, 3] = c5ColorOpticalFrame_to_c5Link_pos

    c1Link_to_c5ColorOpticalFrame = np.dot(c1Link_to_c1ColorOpticalFrame, tf.transformations.inverse_matrix(c1_to_c5_mat))
    c1Link_to_c5Link = np.dot(c1Link_to_c5ColorOpticalFrame, c5ColorOpticalFrame_to_c5Link)

    c1Link_to_c5Link_pos = c1Link_to_c5Link[:3, 3]
    c1Link_to_c5Link_quat = tf.transformations.quaternion_from_matrix(c1Link_to_c5Link)

    # .................. for camera6 to camera1 ..................................... #
    c1_to_c6_mat = np.asarray([[0.35428684, -0.78506002,  0.50809605, -0.13837108],
       [ 0.82388472,  0.51907469,  0.22754217, -0.13304891],
       [-0.44237406,  0.33799738,  0.83070029,  0.03065303],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    c6ColorOpticalFrame_to_c6Link_pos = np.asarray([0.015, 0.000, 0.000])
    c6ColorOpticalFrame_to_c6Link_rot = tf.transformations.quaternion_matrix(np.asarray([0.505, -0.497, 0.503, 0.495]))
    c6ColorOpticalFrame_to_c6Link = c6ColorOpticalFrame_to_c6Link_rot
    c6ColorOpticalFrame_to_c6Link[:3, 3] = c6ColorOpticalFrame_to_c6Link_pos

    c1Link_to_c6ColorOpticalFrame = np.dot(c1Link_to_c1ColorOpticalFrame, tf.transformations.inverse_matrix(c1_to_c6_mat))
    c1Link_to_c6Link = np.dot(c1Link_to_c6ColorOpticalFrame, c6ColorOpticalFrame_to_c6Link)

    c1Link_to_c6Link_pos = c1Link_to_c6Link[:3, 3]
    c1Link_to_c6Link_quat = tf.transformations.quaternion_from_matrix(c1Link_to_c6Link)

    from IPython import embed; embed()

if __name__ == '__main__':
    main()
