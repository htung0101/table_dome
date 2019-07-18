import os
from open3d import *
import numpy as np

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
    save_dir = '/home/zhouxian/newCalibrate/check/points_in_artag'
    pts = np.load(os.path.join(save_dir, 'downsampled_points_cam1inartag_new.npy'))
    print(pts.shape)

    inlier_idxs = get_inlier_idxs(pts)
    pts = pts[inlier_idxs, :]

    pcd = PointCloud()
    pcd.points = Vector3dVector(pts)

    pts1 = np.load(os.path.join(save_dir, 'downsampled_points_cam2inartag_new.npy'))
    inlier_idxs1 = get_inlier_idxs(pts1)
    pts1 = pts1[inlier_idxs1, :]
    pcd1 = PointCloud()
    pcd1.points = Vector3dVector(pts1)

    pts2 = np.load(os.path.join(save_dir, 'downsampled_points_cam3inartag_new.npy'))
    inlier_idxs2 = get_inlier_idxs(pts2)
    pts2 = pts2[inlier_idxs2, :]
    pcd2 = PointCloud()
    pcd2.points = Vector3dVector(pts2)

    pts3 = np.load(os.path.join(save_dir, 'downsampled_points_cam4inartag_new.npy'))
    inlier_idxs3 = get_inlier_idxs(pts3)
    pts3 = pts3[inlier_idxs3, :]
    pcd3 = PointCloud()
    pcd3.points = Vector3dVector(pts3)

    pts4 = np.load(os.path.join(save_dir, 'downsampled_points_cam5inartag_new.npy'))
    inlier_idxs4 = get_inlier_idxs(pts4)
    pts4 = pts4[inlier_idxs4, :]
    pcd4 = PointCloud()
    pcd4.points = Vector3dVector(pts4)

    pts5 = np.load(os.path.join(save_dir, 'downsampled_points_cam6inartag_new.npy'))
    inlier_idxs5 = get_inlier_idxs(pts5)
    pts5 = pts5[inlier_idxs5, :]
    pcd5 = PointCloud()
    pcd5.points = Vector3dVector(pts5)

    frame = geometry.create_mesh_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )

    #draw_geometries([pcd1, pcd, pcd2, pcd3, pcd4, pcd5, frame])
    draw_geometries([pcd1, pcd2, pcd3])

    from IPython import embed; embed()

if __name__ == '__main__':
    main()
