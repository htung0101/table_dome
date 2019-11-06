import os


import os, sys, glob, pickle
import cv2
import numpy as np
import tensorflow as tf
import scipy.misc
import utils_py
import argparse

import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pathos.pools as pp
import ipdb
st = ipdb.set_trace

from itertools import permutations, combinations

sync_dict_keys = [
    'colorIndex1', 'colorIndex2', 'colorIndex3', 'colorIndex4', 'colorIndex5', 'colorIndex6', \
    'depthIndex1', 'depthIndex2', 'depthIndex3', 'depthIndex4', 'depthIndex5', 'depthIndex6', \
]

EPS = 1e-6

MAX_DEPTH_PTS = 200000
MIN_DEPTH_RANGE = 0.05
MAX_DEPTH_RANGE = 0.5

empty_table = True

H = int(480/2.0)
W = int(640/2.0)

MOD = 'aa' # first try
MOD = 'ab' # move the origin back a bit
MOD = "ac" # data collected from 14 videos till newData5 missing(newData3,newData4 and newData1)
MOD = "ba" #added empty option

def process_rgbs(rgb_path):
    utils_py.assert_exists(rgb_path)
    rgb = np.load(rgb_path)
    H_, W_, _ = rgb.shape
    assert(H_==480) # otw i don't know what data this is
    assert(W_==640) # otw i don't know what data this is
    # scale down
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)
    return rgb,H_,W_

def process_depths(depth_path):
    utils_py.assert_exists(depth_path)
    depth = np.load(depth_path)
    depth = depth/1000.0
    depth = depth.astype(np.float32)
    return depth

def process_xyz(depth,pix_T_cam):
    ## pointcloud
    xyz = utils_py.depth2pointcloud_py(depth, pix_T_cam)
    # only take points >5cm ahead of the cam
    xyz = xyz[xyz[:,2] > MIN_DEPTH_RANGE]
    # only take points <1m ahead of the cam
    xyz = xyz[xyz[:,2] < MAX_DEPTH_RANGE]

    if xyz.shape[0] > MAX_DEPTH_PTS:
        np.random.shuffle(xyz)
        xyz = xyz[0:MAX_DEPTH_PTS]
    elif xyz.shape[0] < MAX_DEPTH_PTS:
        xyz = np.pad(xyz, [(0, MAX_DEPTH_PTS - xyz.shape[0]), (0,0)],
                     mode='constant', constant_values=0)
    return xyz

def job(data_dir, save_dir):

    #data_dir = '/projects/katefgroup/gauravp/{}'.format(folderName)
    #st()
    color_dir = '%s/colorData' % data_dir
    depth_dir = '%s/depthData' % data_dir

    sync_idx_file = '%s/syncedIndexData.pkl' % data_dir
    sync_data = utils_py.load_from_pkl(sync_idx_file)
    # sync_data indicates what frame to grab from each cam at each timestep    

    nFrames = len(sync_data[sync_dict_keys[0]])

    # it is suspicious that this file is so old, but OK let's try...
    camera_info_file = '/projects/katefgroup/gauravp/all_data_with_intrinsics.pkl'
    camera_info = utils_py.load_from_pkl(camera_info_file)

    ar_T_cam0 = np.linalg.inv(camera_info['ar_in_cam_0'])
    # we want y to point downward (following computer vision conventions)
    # also, let's move it up 0.05m and back 0.4m, 
    # so that the pointcloud distr post transformation is somewhat similar

    origin_T_ar = np.array([[0, 1.0, 0, 0],
                           [0, 0, -1.0, 0.05],
                           [1.0, 0, 0, 0.4],
                           [0, 0, 0, 1]],
                          dtype=np.float32)

    origin_T_cam0 = np.matmul(origin_T_ar, ar_T_cam0)

    # for frame_ind in range(nFrames):
    # for frame_ind in [1100]:
    # for frame_ind in range(0, nFrames, 100):
    # for frame_ind in range(0, 1000, 50):
    for frame_ind in range(0, nFrames, 10):
        all_depths = []
        all_rgb_camXs = []
        all_origin_T_camXs = []
        all_pix_T_cams = []
        all_xyz_camXs = []

        one_empty_rgb_camXs = []
        one_empty_xyz_camXs = []

        for cam_ind in range(1,7):
            ## rgb
            rgb_cam_name = 'colorIndex%d' % cam_ind
            frame_id = sync_data[rgb_cam_name][frame_ind]
            rgb_path = '%s/colorData/Cam%d/cam_camera%d_%d_color.npy' % (data_dir, cam_ind, cam_ind, frame_id)
            rgb,H_,W_ = process_rgbs(rgb_path)
            
            all_rgb_camXs.append(rgb)
            if empty_table:
                rgb_path_empty = '%s/colorData/Cam%d/cam_camera%d_%d_color.npy' % (data_dir, cam_ind, cam_ind, 0)
                rgb_empty,_,_ = process_rgbs(rgb_path_empty)
                one_empty_rgb_camXs.append(rgb_empty)

            ## depth
            depth_cam_name = 'depthIndex%d' % cam_ind
            frame_id = sync_data[depth_cam_name][frame_ind]
            depth_path = '%s/depthData/Cam%d/cam_camera%d_%d_depth.npy' % (data_dir, cam_ind, cam_ind, frame_id)
            depth = process_depths(depth_path)
            all_depths.append(depth)
            
            if empty_table:
                depth_path_empty = '%s/depthData/Cam%d/cam_camera%d_%d_depth.npy' % (data_dir, cam_ind, cam_ind, 0)
                depth_empty = process_depths(depth_path_empty)

            

            ## intrinsics
            # it seems that the intrinsics are ZERO-indexed
            pix_T_cam = camera_info['cam_%d_intrinsics' % (cam_ind-0)]
            # we want to scale this down, but only after we unproject the depth...
            
            ## extrinsics
            # it seems that the extrinsics are ONE-indexed
            if cam_ind==1:
                cam0_T_camX = np.eye(4)
            else:
                cam0_T_camX = np.linalg.inv(camera_info['cam_0_to_%d' % (cam_ind-1)])

            origin_T_camX = np.matmul(origin_T_cam0, cam0_T_camX)
            all_origin_T_camXs.append(origin_T_camX.astype(np.float32))
            
            xyz = process_xyz(depth,pix_T_cam)
            all_xyz_camXs.append(xyz.astype(np.float32))

            if empty_table:
                xyz_empty = process_xyz(depth_empty,pix_T_cam)
                one_empty_xyz_camXs.append(xyz_empty.astype(np.float32))

            # scale down intrinsics
            sx = W/float(W_)
            sy = H/float(H_)
            pix_T_cam = utils_py.scale_projection_mat(pix_T_cam, sx, sy)
            all_pix_T_cams.append(pix_T_cam.astype(np.float32))
            
        print('got all data from {} for frame {:05d}'.format(folderName,frame_ind))

        all_cams = [0, 1, 3, 4, 5]
        # we can output all permutations
        combos = list(permutations(all_cams, 2))
        cams_to_use = []
        for cam_set in combos:
            cams_to_use.append(cam_set)
        # print cams_to_use

        out_dir_base = '/projects/katefgroup/datasets/table'
        out_dir = '%s/%s' % (out_dir_base, MOD)
        utils_py.mkdir(out_dir)

        for cams in cams_to_use:
            # print 'writing tfr for these cams:', cams

            ## gen a filename
            out_fn = '{}_{:05d}'.format(folderName,frame_ind)

            for cam in cams:
                out_fn += '_cam%s' % cam
            out_fn += '.tfrecord'
            out_f = os.path.join(out_dir, out_fn)
            if os.path.isfile(out_f):
                sys.stdout.write(':')
            else:
                # grab the subset of cams
                pix_T_cams_ = []
                rgb_camXs_ = []
                xyz_camXs_ = []
                origin_T_camXs_ = []
                if empty_table:
                    empty_rgb_camXs_ = []
                    empty_xyz_camXs_ = []
                for cam in cams:
                    pix_T_cams_.append(all_pix_T_cams[cam])
                    origin_T_camXs_.append(all_origin_T_camXs[cam])
                    rgb_camXs_.append(all_rgb_camXs[cam])
                    xyz_camXs_.append(all_xyz_camXs[cam])
                    if empty_table:
                        empty_rgb_camXs_.append(one_empty_rgb_camXs[cam])
                        empty_xyz_camXs_.append(one_empty_xyz_camXs[cam])
                    
                pix_T_cams = np.stack(pix_T_cams_, axis=0)
                rgb_camXs = np.stack(rgb_camXs_, axis=0)
                xyz_camXs = np.stack(xyz_camXs_, axis=0)
                origin_T_camXs = np.stack(origin_T_camXs_, axis=0)
    
                if empty_table:
                    empty_rgb_camXs = np.stack(empty_rgb_camXs_, axis=0)
                    empty_xyz_camXs = np.stack(empty_xyz_camXs_, axis=0)


                assert rgb_camXs.dtype == np.uint8
                assert xyz_camXs.dtype == np.float32
                # st()
                if empty_table:
                    # st()
                    assert empty_rgb_camXs.dtype == np.uint8
                    assert empty_xyz_camXs.dtype == np.float32
                
                assert origin_T_camXs.dtype == np.float32
                assert pix_T_cams.dtype == np.float32

                rgb_camXs_raw = rgb_camXs.tostring()
                xyz_camXs_raw = xyz_camXs.tostring()
                origin_T_camXs_raw = origin_T_camXs.tostring()
                pix_T_cams_raw = pix_T_cams.tostring()

                if empty_table:
                    empty_rgb_camXs_raw = empty_rgb_camXs.tostring()
                    empty_xyz_camXs_raw = empty_xyz_camXs.tostring()


                """
                comptype = tf.python_io.TFRecordCompressionType.GZIP
                compress = tf.python_io.TFRecordOptions(compression_type=comptype)
                writer = tf.python_io.TFRecordWriter(out_f, options=compress)
                feature = {
                    'filename' : utils_py.bytes_feature(out_fn),
                    'pix_T_cams_raw' : utils_py.bytes_feature(pix_T_cams_raw),
                    'origin_T_camXs_raw' : utils_py.bytes_feature(origin_T_camXs_raw),
                    'rgb_camXs_raw' : utils_py.bytes_feature(rgb_camXs_raw),
                    'xyz_camXs_raw' : utils_py.bytes_feature(xyz_camXs_raw),

                    'empty_rgb_camXs_raw':utils_py.bytes_feature(empty_rgb_camXs_raw),
                    'empty_xyz_camXs_raw':utils_py.bytes_feature(empty_xyz_camXs_raw)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                writer.close()
                """



                sys.stdout.write('.')
            sys.stdout.flush()
        # done cams
    # done frames
    print('done')

def main():
    # folders = ["Data2","Data3","Data4","Data5","Data6","Data7","Data9","Data10","Data11","Data12","Data13","newData1","newData2","newData5"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True,\
        help='specify the directory you want to save data to')
    parser.add_argument('--data_dir', type=str, required=True,\
        help='for which camera')
    args = parser.parse_args()
    job(args.data_dir, args.save_dir)


if __name__ == '__main__':
    main()
    # python packtf.py --save_dir