import pickle
import os
import yaml
import numpy as np
import argparse
import pdb
# import pyquaternion
import tf.transformations as tft

def main():
    n_cam = 6
    extrinsics_pkl_folder = '/home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m11_d6_h19_m4_s46/ar_tag'
    ar_T_camXs = []
    for i in range(n_cam):
        ar_T_camXs.append(np.linalg.inv(pickle.load(open(os.path.join(extrinsics_pkl_folder, 'camera{}_ar_tag.pkl'.format(i+1)), 'rb'))))

    # all the link transforms
    cam0coloroptical_frame_T_cam0link = np.eye(4)
    cam0coloroptical_frame_T_cam0link[:3, 3] = np.asarray([0.015, 0.001, 0.000])
    cam0coloroptical_frame_T_cam0link[:3, :3] = tft.quaternion_matrix([0.506, -0.494, 0.507, 0.492])[:3, :3]
    # cam0coloroptical_frame_T_cam0link[:3, :3] = pyquaternion.Quaternion([0.492, 0.506, -0.494, 0.507]).rotation_matrix

    cam1coloroptical_frame_T_cam1link = np.eye(4)
    cam1coloroptical_frame_T_cam1link[:3, 3] = np.asarray([0.015, 0.001, -0.000])
    cam1coloroptical_frame_T_cam1link[:3, :3] = tft.quaternion_matrix([0.507, -0.495, 0.501, 0.497])[:3, :3]
    # cam1coloroptical_frame_T_cam1link[:3, :3] = pyquaternion.Quaternion([0.497, 0.507, -0.495, 0.501]).rotation_matrix

    cam2coloroptical_frame_T_cam2link = np.eye(4)
    cam2coloroptical_frame_T_cam2link[:3, 3] = np.asarray([0.015, 0.001, 0.000])
    cam2coloroptical_frame_T_cam2link[:3, :3] = tft.quaternion_matrix([0.505, -0.495, 0.504, 0.495])[:3, :3]
    # cam2coloroptical_frame_T_cam2link[:3, :3] = pyquaternion.Quaternion([0.495, 0.505, -0.495, 0.504]).rotation_matrix

    cam3coloroptical_frame_T_cam3link = np.eye(4)
    cam3coloroptical_frame_T_cam3link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam3coloroptical_frame_T_cam3link[:3, :3] = tft.quaternion_matrix([0.506, -0.495, 0.504, 0.495])[:3, :3]
    # cam3coloroptical_frame_T_cam3link[:3, :3] = pyquaternion.Quaternion([0.495, 0.506, -0.495, 0.504]).rotation_matrix

    cam4coloroptical_frame_T_cam4link = np.eye(4)
    cam4coloroptical_frame_T_cam4link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam4coloroptical_frame_T_cam4link[:3, :3] = tft.quaternion_matrix([0.505, -0.496, 0.500, 0.499])[:3, :3]
    # cam4coloroptical_frame_T_cam4link[:3, :3] = pyquaternion.Quaternion([0.499, 0.505, -0.496, 0.500]).rotation_matrix

    cam5coloroptical_frame_T_cam5link = np.eye(4)
    cam5coloroptical_frame_T_cam5link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam5coloroptical_frame_T_cam5link[:3, :3] = tft.quaternion_matrix([0.505, -0.497, 0.503, 0.495])[:3, :3]
    # cam5coloroptical_frame_T_cam5link[:3, :3] = pyquaternion.Quaternion([0.495, 0.505, -0.497, 0.503]).rotation_matrix
    import IPython;IPython.embed()
    ar_T_camXslink = []
    for i in range(n_cam):
        ar_T_camXslink.append(np.dot(ar_T_camXs[i], eval('cam{0}coloroptical_frame_T_cam{1}link'.format(i,i))))

    for i in range(n_cam):
        pos = ar_T_camXslink[i][:3, 3]
        # quat = pyquaternion.Quaternion(matrix=ar_T_camXslink[i][:3, :3])
        euler = tft.euler_from_matrix(ar_T_camXslink[i][:3, :3], 'rzyx')
        
        command = 'rosrun tf static_transform_publisher {0} {1} {2} {3} {4} {5} ref_frame camera{6}_link 20'.format(pos[0], pos[1], pos[2], euler[0], euler[1], euler[2], i+1)
        print(command)

if __name__ == '__main__':
    main()