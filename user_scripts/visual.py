import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
import pickle
import pcl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import tf
import argparse
from open3d import *
import ipdb
st=ipdb.set_trace

def parse_intrinsics(intrinsic_parameters):
    """
    parameters:
    -------------
        intrinsic_parameters: a 3x3 matrix containing intrinsic parameters
    returns:
    -------------
        fx, fy, s, cx, cy
    """
    fx = intrinsic_parameters[0][0]
    fy = intrinsic_parameters[1][1]
    cx = intrinsic_parameters[0][2]
    cy = intrinsic_parameters[1][2]
    s = intrinsic_parameters[0][1]
    return fx, fy, s, cx, cy


def unproject_(depth_img, depth_cam_intrinsics,
    from_numpy=True, using_cropped=False):
    """
    for each pixel in depth image use the unprojection formula to unproject
    requires depth image and camera intrinsics.
    """
    points = list()
    fx, fy, s, cx, cy = parse_intrinsics(depth_cam_intrinsics)
    if using_cropped:
        # TODO: can you compute this automatically somehow
        print('you should not be using this')
        from IPython import embed; embed()
        cx = 230.
        cy = 230.

    # from IPython import embed; embed()
    # NOTE: this means you will have to load the file as numpy array
    if not from_numpy:
        depth_img = np.load(depth_img).squeeze(2)
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            depth_xy = depth_img[i][j] / 1000.
            px = (j - cx) * (depth_xy) / fx
            py = (i - cy) * (depth_xy) / fy
            pz = depth_xy
            points.append(np.asarray([px, py, pz]))
    points = np.stack(points)
    return points


def plot_points(points, cam_no=None, fig=None):
    if not fig:
        print('not plotting')
        return
    fig.scatter(points[:,0], points[:, 1], points[:,2], s=0.1)
    plt.show()

def quat_to_matrix(quat_dict):
    pos = quat_dict["position"]
    quat = quat_dict["orientation"]

    pose_matrix = tf.transformations.quaternion_matrix(quat)
    pose_matrix[0:3, 3] = pos
    return pose_matrix

def get_inlier_idxs(pts):
    """
    parameters:
    -------------
        pts: (N, 3) array of points
    returns:
    ------------
        idxs: indexes of points which are inside the sphere of radius 0.5
    """
    radius_sq = 0.5 ** 2
    pts_norm_sq = np.linalg.norm(pts, axis=1)
    pts_norm_sq = pts_norm_sq - radius_sq
    return pts_norm_sq <= 0.0

def main():

    # NOTE: this is only used to get the camera extrinsics and color img which is
    # not even required as of now.
    data_path = os.path.join(config.data_root, config.record_name)
    vr_data_path = os.path.join(data_path, "vr_tag")
    num_cam = config.NUM_CAM

    intrinsic_f = open(os.path.join(vr_data_path, "intrinsics.pkl"), "rb")
    intrinsic_mat =  pickle.load(intrinsic_f)
    frame_id = 10
    pcd_list = []
    for cam_id in range(1, 1+num_cam):
        vr_tag_f = open(os.path.join(vr_data_path, "camera{}_vr_tag.pkl".format(cam_id)), "rb")
        artag_T_camX = quat_to_matrix(pickle.load(vr_tag_f))
        camX_T_artag = tf.transformations.inverse_matrix(artag_T_camX)
        intrinsic_camX = intrinsic_mat[str(cam_id)]


        # read depth image
        depth_data_file = os.path.join(data_path, "rgb_depth_npy/depthData/Cam{}/cam_{}_depth.npy".format(cam_id, cam_id))
        depth_image = np.load(depth_data_file)[frame_id, ...]
        points = unproject_(depth_image,\
                intrinsic_camX,
                from_numpy=True, using_cropped=False)
        points = np.c_[points, np.ones(points.shape[0])]
        #points_from_camera['cam_{}'.format(cam_id)] = points
        p_artag = np.dot(camX_T_artag, points.T).T

        p_artag = p_artag[:, :3].astype(np.float32)

        p = pcl.PointCloud(p_artag)
        downsample_points_filter = p.make_voxel_grid_filter()
        downsample_points_filter.set_leaf_size(0.01, 0.01, 0.01)
        pts = downsample_points_filter.filter()
        pts = np.asarray(pts)

        inlier_idxs = get_inlier_idxs(pts)
        pts = pts[inlier_idxs, :]

        pcd = PointCloud()
        pcd.points = Vector3dVector(pts)
        pcd_list.append(pcd)

    frame = geometry.create_mesh_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    draw_geometries(pcd_list)

    #plt.show()


if __name__ == '__main__':
    main()