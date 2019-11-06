import pickle
import os
import yaml
import numpy as np
import argparse
import pdb
import pyquaternion

def main():
    # cam4_T_cam3 = np.load('cam4_T_cam3.npy')
    cam4_T_cam3 = np.load('/home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m11_h16_m27_s25/ar_tag/ar_tag_cam4_T_cam3.npy')


    # all the link transforms
    cam3coloroptical_frame_T_cam3link = np.eye(4)
    cam3coloroptical_frame_T_cam3link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam3coloroptical_frame_T_cam3link[:3, :3] = pyquaternion.Quaternion([0.495, 0.506, -0.495, 0.504]).rotation_matrix

    cam4coloroptical_frame_T_cam4link = np.eye(4)
    cam4coloroptical_frame_T_cam4link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam4coloroptical_frame_T_cam4link[:3, :3] = pyquaternion.Quaternion([0.499, 0.505, -0.496, 0.500]).rotation_matrix


    cam3link_T_cam4 = np.dot(np.linalg.inv(cam3coloroptical_frame_T_cam3link),
            np.linalg.inv(cam4_T_cam3))
    cam3link_T_cam4link = np.dot(cam3link_T_cam4, cam4coloroptical_frame_T_cam4link)

    pos = cam3link_T_cam4link[:3, 3]
    quat = pyquaternion.Quaternion(matrix=cam3link_T_cam4link[:3, :3])
    print(pos)
    print(quat.elements)

    command = f'rosrun tf static_transform_publisher {pos[0]} {pos[1]} {pos[2]} {quat[1]} {quat[2]} {quat[3]} {quat[0]} camera3_link camera4_link 100'
    print('running command: ', command)
    os.system(command)
    # pdb.set_trace()


if __name__ == '__main__':
    main()