import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import pickle


'''
P3D.x = (x_d - cx_d) * depth(x_d,y_d) / fx_d
P3D.y = (y_d - cy_d) * depth(x_d,y_d) / fy_d
P3D.z = depth(x_d,y_d)
'''

def main():
    # Load .pkl file
    # Load images from the indices given in the .pkl file
    # Unproject images
    # Dump pointcloud data into a .pkl file or plot it
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices_path",type=str,required=True)
    args=parser.parse_args()

    if not os.path.exists(args.indices_path):
        print("Invalid Path.\n")
        os.sys.exit(-1)

    with open(args.indices_path, "rb") as f:
        data = pickle.load(f)

    depthImgPath = '/home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData'
    colorImgPath = '/home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData'

    #try reconstruction for one set of images from each of the cameras.
    depth1 = np.load(depthImgPath + '/Cam1/cam_camera1_1_depth.npy', allow_pickle=True)
    depth2 = np.load(depthImgPath + '/Cam2/cam_camera2_1_depth.npy', allow_pickle=True)
    depth3 = np.load(depthImgPath + '/Cam3/cam_camera3_1_depth.npy', allow_pickle=True)
    depth4 = np.load(depthImgPath + '/Cam4/cam_camera4_1_depth.npy', allow_pickle=True)
    depth5 = np.load(depthImgPath + '/Cam5/cam_camera5_1_depth.npy', allow_pickle=True)
    depth6 = np.load(depthImgPath + '/Cam6/cam_camera6_1_depth.npy', allow_pickle=True)

    with open('all_data_with_depth_intrinsics.pkl',"rb") as f:
        all_data_with_intrinsics = pickle.load(f)



    '''

    ['depthIndex3', 'depthIndex2', 'colorIndex6', 'depthIndex1', 'colorIndex4', 'colorIndex5', 'colorIndex2', 'colorIndex3', 'depthIndex6', 'colorIndex1', 'depthIndex4', 'depthIndex5']

    '''

    import pdb;pdb.set_trace()


if __name__ == "__main__":
    main()