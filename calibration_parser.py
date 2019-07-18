import pickle
import os
import yaml
import numpy as np
import argparse
import pdb
import transformations

def main():
    # Need pose in camera_1 pose in camera_2
    # and need the transforms between the camera
    # path to yaml file containing transformations

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file_path', type=str, required=True)
    # parser.add_argument('--pose_fileCam1', type=str, required=True)
    # parser.add_argument('--pose_fileCam2', type=str, required=True)

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

    # if not os.path.exists(args.pose_fileCam1):
    #     print('marker pose 1 file not found')
    #     os.sys.exit(-1)

    # if not os.path.exists(args.pose_fileCam2):
    #     print('marker pose 2 file not found')
    #     os.sys.exit(-1)

    # with open(args.pose_fileCam1, 'rb') as pose_1_handler:
    #     pose1_dict = pickle.load(pose_1_handler)

    # pose_1_handler.close()

    # with open(args.pose_fileCam2, 'rb') as pose_2_handler:
    #     pose2_dict = pickle.load(pose_2_handler)

    # pose_2_handler.close()

    #pos_cam_0 = np.concatenate((pose1_dict['position'], np.ones(1)))
    # pos2 = np.concatenate((pose2_dict['position'], np.ones(1)))

    cam_0_to_1 = T_cn_cnm1['cam1']
    cam_1_to_2 = T_cn_cnm1['cam2']
    cam_2_to_3 = T_cn_cnm1['cam3']
    cam_3_to_4 = T_cn_cnm1['cam4']
    cam_4_to_5 = T_cn_cnm1['cam5']

    cam_0_to_2 = np.dot(cam_1_to_2, cam_0_to_1)
    cam_0_to_3 = np.dot(cam_2_to_3, cam_0_to_2)
    cam_0_to_4 = np.dot(cam_3_to_4, cam_0_to_3)
    cam_0_to_5 = np.dot(cam_4_to_5, cam_0_to_4)

    # pos_cam_1 = np.dot(cam_0_to_1, pos_cam_0)
    # pos_cam_2 = np.dot(cam_0_to_2, pos_cam_0)
    # pos_cam_3 = np.dot(cam_0_to_3, pos_cam_0)
    # pos_cam_4 = np.dot(cam_0_to_4, pos_cam_0)
    # pos_cam_5 = np.dot(cam_0_to_5, pos_cam_0)

    #import pdb;pdb.set_trace()
    save_dict = {}
    save_dict['cam_0_to_1'] = cam_0_to_1
    save_dict['cam_0_to_2'] = cam_0_to_2
    save_dict['cam_0_to_3'] = cam_0_to_3
    save_dict['cam_0_to_4'] = cam_0_to_4
    save_dict['cam_0_to_5'] = cam_0_to_5

    # cam_1_to_0 = np.linalg.inv(cam_0_to_1)
    # cam_2_to_0 = np.linalg.inv(cam_0_to_2)
    # cam_3_to_0 = np.linalg.inv(cam_0_to_3)
    # cam_4_to_0 = np.linalg.inv(cam_0_to_4)
    # cam_5_to_0 = np.linalg.inv(cam_0_to_5)

    # TODO: Amazingly bad code.
    ar_marker_in_cam0_pos = np.asarray([0.0341482351409, -0.00832891337583, 0.369164266858])
    quat_pose = np.asarray([-0.219903190751, -0.660172128305, 0.671450363539, -0.254891657395])

    pose_matrix = transformations.quaternion_matrix(quat_pose)
    pose_matrix[0:3, 3] = ar_marker_in_cam0_pos

    save_dict['ar_in_cam_0'] = pose_matrix
    # camera coordinate frames in world coordinates.
    # cam_0_in_ar = np.linalg.inv(pose_matrix)
    # cam_1_in_ar = np.dot(cam_1_to_0, cam_0_in_ar)
    # cam_2_in_ar = np.dot(cam_2_to_0, cam_0_in_ar)
    # cam_3_in_ar = np.dot(cam_3_to_0, cam_0_in_ar)
    # cam_4_in_ar = np.dot(cam_4_to_0, cam_0_in_ar)
    # cam_5_in_ar = np.dot(cam_5_to_0, cam_0_in_ar)

    # save_dict['cam_0_in_ar'] = cam_0_in_ar
    # save_dict['cam_1_in_ar'] = cam_1_in_ar
    # save_dict['cam_2_in_ar'] = cam_2_in_ar
    # save_dict['cam_3_in_ar'] = cam_3_in_ar
    # save_dict['cam_4_in_ar'] = cam_4_in_ar
    # save_dict['cam_5_in_ar'] = cam_5_in_ar

    with open('all_data_new_calib.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
    f.close()

    pdb.set_trace()


if __name__ == '__main__':
    main()
