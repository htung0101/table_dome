import pickle
import os
import yaml
import numpy as np
import argparse
import pdb
import pyquaternion
import math3d as m3d

def main():
    # Need pose in camera_1 pose in camera_2
    # and need the transforms between the camera
    # path to yaml file containing transformations

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.yaml_file_path):
        print('not found')
        os.sys.exit(-1)

    with open(args.yaml_file_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
        T_cn_cnm1=dict()
        for i in yaml_data:
            if 'T_cn_cnm1' in yaml_data[i]:
                T_cn_cnm1[i] = np.array(yaml_data[i]['T_cn_cnm1'])

    cam1_T_cam0 = T_cn_cnm1['cam1']
    cam2_T_cam1 = T_cn_cnm1['cam2']
    cam3_T_cam2 = T_cn_cnm1['cam3']
    cam4_T_cam3 = T_cn_cnm1['cam4']
    cam5_T_cam4 = T_cn_cnm1['cam5']

    cam1_T_cam0 = cam1_T_cam0
    cam2_T_cam0 = np.dot(cam2_T_cam1, cam1_T_cam0)
    cam3_T_cam0 = np.dot(cam3_T_cam2, cam2_T_cam0)
    cam4_T_cam0 = np.dot(cam4_T_cam3, cam3_T_cam0)
    cam5_T_cam0 = np.dot(cam5_T_cam4, cam4_T_cam0)


    # all the link transforms
    cam0coloroptical_frame_T_cam0link = np.eye(4)
    cam0coloroptical_frame_T_cam0link[:3, 3] = np.asarray([0.015, 0.001, 0.000])
    cam0coloroptical_frame_T_cam0link[:3, :3] = pyquaternion.Quaternion([0.492, 0.506, -0.494, 0.507]).rotation_matrix

    cam1coloroptical_frame_T_cam1link = np.eye(4)
    cam1coloroptical_frame_T_cam1link[:3, 3] = np.asarray([0.015, 0.001, -0.000])
    cam1coloroptical_frame_T_cam1link[:3, :3] = pyquaternion.Quaternion([0.497, 0.507, -0.495, 0.501]).rotation_matrix

    cam2coloroptical_frame_T_cam2link = np.eye(4)
    cam2coloroptical_frame_T_cam2link[:3, 3] = np.asarray([0.015, 0.001, 0.000])
    cam2coloroptical_frame_T_cam2link[:3, :3] = pyquaternion.Quaternion([0.495, 0.505, -0.495, 0.504]).rotation_matrix

    cam3coloroptical_frame_T_cam3link = np.eye(4)
    cam3coloroptical_frame_T_cam3link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam3coloroptical_frame_T_cam3link[:3, :3] = pyquaternion.Quaternion([0.495, 0.506, -0.495, 0.504]).rotation_matrix

    cam4coloroptical_frame_T_cam4link = np.eye(4)
    cam4coloroptical_frame_T_cam4link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam4coloroptical_frame_T_cam4link[:3, :3] = pyquaternion.Quaternion([0.499, 0.505, -0.496, 0.500]).rotation_matrix

    cam5coloroptical_frame_T_cam5link = np.eye(4)
    cam5coloroptical_frame_T_cam5link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam5coloroptical_frame_T_cam5link[:3, :3] = pyquaternion.Quaternion([0.495, 0.505, -0.497, 0.503]).rotation_matrix

    cam0link_T_camXslink = list()
    for i in range(5):
        cam0link_T_camX = np.dot(np.linalg.inv(cam0coloroptical_frame_T_cam0link),
            np.linalg.inv(eval(f'cam{i+1}_T_cam0')))
        cam0link_T_camXlink = np.dot(cam0link_T_camX,
            eval(f'cam{i+1}coloroptical_frame_T_cam{i+1}link'))

        cam0link_T_camXslink.append(cam0link_T_camXlink)

    cam0link_T_camXslink = np.stack(cam0link_T_camXslink)

    for i in range(1, len(cam0link_T_camXslink)+1):
        pos = cam0link_T_camXslink[i-1][:3, 3]
        quat = pyquaternion.Quaternion(matrix=cam0link_T_camXslink[i-1][:3, :3])
        print(pos)
        print(quat.elements)

        command = f'rosrun tf static_transform_publisher {pos[0]} {pos[1]} {pos[2]} {quat[1]} {quat[2]} {quat[3]} {quat[0]} camera{1}_link camera{i+1}_link 100'
        print(command)

    pdb.set_trace()


if __name__ == '__main__':
    main()