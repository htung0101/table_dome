import numpy as np
import yaml
import argparse
import os
import pickle
from scipy.spatial.distance import cdist

'''
    COMMAND FOR RUNNING THE CODE:
    python timeCompare.py --path /projects/katefgroup/spshetty/NewData/TableDome_y2019_m8_h10_m42_s7_ab/rgb_depth_npy
    python timeCompareGeorge.py --path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m10_h0_m32_s33/rgb_depth_npy
'''


def time2int(t):
    return t.secs * 10 ** 9 + t.nsecs


def main():
    NUM_CAMS = 6
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    color_paths = {}
    depth_paths = {}
    color_data = {}
    depth_data = {}

    for i in range(1, NUM_CAMS + 1):
        color_paths['colorIndex' + str(i)] = args.path + '/colorData/Cam' + str(i) + '/data.yaml'
        if not os.path.exists(color_paths['colorIndex' + str(i)]):
            print("Invalid Path for color Image File " + str(i))
            os.sys.exit(-1)
        else:
            with open(color_paths['colorIndex' + str(i)], 'r') as f:
                color_data['colorIndex' + str(i)] = np.array([time2int(t['Timestamp']) for t in yaml.load(f).values()], dtype=np.int64)

        depth_paths['depthIndex' + str(i)] = args.path + '/depthData/Cam' + str(i) + '/data.yaml'
        if not os.path.exists(depth_paths['depthIndex' + str(i)]):
            print("Invalid Path for depth Image File " + str(i))
            os.sys.exit(-1)
        else:
            with open(depth_paths['depthIndex' + str(i)], 'r') as f:
                depth_data['depthIndex' + str(i)] = np.array([time2int(t['Timestamp']) for t in yaml.load(f).values()], dtype=np.int64)

    # Camera with minimum number of frames
    reference_cam = min(color_data, key=lambda x: color_data[x].shape[0])

    index_data = {}
    for k, v in color_data.items():
        if k != reference_cam:
            distance_matrix = cdist(color_data[reference_cam].reshape(-1, 1), v.reshape(-1, 1), 'cityblock')
            closest_index = np.argmin(distance_matrix, axis=1)
            index_data[k] = closest_index
        else:
            index_data[k] = np.array(range(color_data[k].shape[0]))

    for k, v in depth_data.items():
        distance_matrix = cdist(color_data[reference_cam].reshape(-1, 1), v.reshape(-1, 1), 'cityblock')
        closest_index = np.argmin(distance_matrix, axis=1)
        index_data[k] = closest_index

    print("save", os.path.join(args.path, "syncedIndexData.pkl"))
    pickle.dump(index_data, open(os.path.join(args.path, "syncedIndexData.pkl"), "wb"))


if __name__ == "__main__":
    main()