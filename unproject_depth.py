import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
import pcl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import tf

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


def main():
    # NOTE: this is only used to get the camera extrinsics and color img which is
    # not even required as of now.
    all_data_file = open('/home/zhouxian/newCalibrate/all_data_new_calib.pkl', 'rb')
    all_data = pickle.load(all_data_file)

    # camera_intrinsics_fish = os.path.join('/home/zhouxian/resized_depth_images/intrinsics.pkl')
    # with open(camera_intrinsics_fish, 'rb') as f:
    #    intrinsic_fish = pickle.load(f)
    # f.close()
    #import pdb; pdb.set_trace()
    ar_marker_in_cam0 = all_data['ar_in_cam_0']
    cam0_to_artag = tf.transformations.inverse_matrix(ar_marker_in_cam0)
    cam0_to_cam1 = all_data['cam_0_to_1']
    cam1_to_cam0 = tf.transformations.inverse_matrix(cam0_to_cam1)
    cam0_to_cam2 = all_data['cam_0_to_2']
    cam2_to_cam0 = tf.transformations.inverse_matrix(cam0_to_cam2)
    cam0_to_cam3 = all_data['cam_0_to_3']
    cam3_to_cam0 = tf.transformations.inverse_matrix(cam0_to_cam3)
    cam0_to_cam4 = all_data['cam_0_to_4']
    cam4_to_cam0 = tf.transformations.inverse_matrix(cam0_to_cam4)
    cam0_to_cam5 = all_data['cam_0_to_5']
    cam5_to_cam0 = tf.transformations.inverse_matrix(cam0_to_cam5)


    # everything in ar_tag transforms
    cam1_to_artag = np.dot(cam0_to_artag, cam1_to_cam0)
    cam2_to_artag = np.dot(cam0_to_artag, cam2_to_cam0)
    cam3_to_artag = np.dot(cam0_to_artag, cam3_to_cam0)
    cam4_to_artag = np.dot(cam0_to_artag, cam4_to_cam0)
    cam5_to_artag = np.dot(cam0_to_artag, cam5_to_cam0)

    # now strategy is for each color and depth image I will first unproject the points
    # then I will use the extrinsics to rotate and bring them all in ar_marker frame
    NUM_CAMERAS = 6
    points_from_camera = {}
    depthImgPath = '/home/zhouxian/newCalibrate/Data1/depthData'
    colorImgPath = '/home/zhouxian/newCalibrate/Data1/colorData'

    '''
    for cam_no in range(NUM_CAMERAS):
        points = unproject_(all_data['depth_img{}'.format(cam_no+1)],\
                all_data['img{}'.format(cam_no+1)],\
                all_data['cam_{}_intrinsics'.format(cam_no+1)],
                from_numpy=True, using_cropped=False)
    '''
    for cam_no in range(NUM_CAMERAS):
        print((cam_no+1))
        #import pdb; pdb.set_trace()
        depth_image=np.load(depthImgPath + '/Cam{}/cam_camera{}_20_depth.npy'.format(cam_no+1,cam_no+1), allow_pickle = True)
        points = unproject_(depth_image,\
                all_data['cam_{}_depth_intrinsics'.format(cam_no+1)],
                from_numpy=True, using_cropped=False)
        
        # first view these points in ar_marker frame and then move ahead
        points = np.c_[points, np.ones(points.shape[0])]
        points_from_camera['cam_{}'.format(cam_no)] = points

    # .... points in cam0 transformed to points in ar_tag .... #
    cam0_points_in_artag = np.dot(cam0_to_artag, points_from_camera['cam_0'].T).T
    cam1_points_in_artag = np.dot(cam1_to_artag, points_from_camera['cam_1'].T).T
    cam2_points_in_artag = np.dot(cam2_to_artag, points_from_camera['cam_2'].T).T
    cam3_points_in_artag = np.dot(cam3_to_artag, points_from_camera['cam_3'].T).T
    cam4_points_in_artag = np.dot(cam4_to_artag, points_from_camera['cam_4'].T).T
    cam5_points_in_artag = np.dot(cam5_to_artag, points_from_camera['cam_5'].T).T
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    cam0_points_in_artag = cam0_points_in_artag[:, :3].astype(np.float32)
    cam1_points_in_artag = cam1_points_in_artag[:, :3].astype(np.float32)
    cam2_points_in_artag = cam2_points_in_artag[:, :3].astype(np.float32)
    cam3_points_in_artag = cam3_points_in_artag[:, :3].astype(np.float32)
    cam4_points_in_artag = cam4_points_in_artag[:, :3].astype(np.float32)
    cam5_points_in_artag = cam5_points_in_artag[:, :3].astype(np.float32)
    # points_in1 = points_from_camera['cam_0']
    # points_in1 = points_in1.astype(np.float32)
    # points_in1 = points_in1[:, :3]

    save_dir = "/home/zhouxian/newCalibrate/check/points_in_artag"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    p = pcl.PointCloud(cam0_points_in_artag)
    downsample_points_filter = p.make_voxel_grid_filter()
    downsample_points_filter.set_leaf_size(0.01, 0.01, 0.01)
    pts = downsample_points_filter.filter()
    downsampled_points = np.asarray(pts)
    print(downsampled_points.shape)
    with open(os.path.join(save_dir, 'downsampled_points_cam1inartag_new.npy'), 'wb') as f:
        np.save(f, downsampled_points)
    f.close()

    p1 = pcl.PointCloud(cam1_points_in_artag)
    downsample_points_filter = p1.make_voxel_grid_filter()
    downsample_points_filter.set_leaf_size(0.01, 0.01, 0.01)
    pts = downsample_points_filter.filter()
    downsampled_points = np.asarray(pts)
    print(downsampled_points.shape)
    with open(os.path.join(save_dir, 'downsampled_points_cam2inartag_new.npy'), 'wb') as f:
        np.save(f, downsampled_points)
    f.close()

    p2 = pcl.PointCloud(cam2_points_in_artag)
    downsample_points_filter = p2.make_voxel_grid_filter()
    downsample_points_filter.set_leaf_size(0.01, 0.01, 0.01)
    pts = downsample_points_filter.filter()
    downsampled_points = np.asarray(pts)
    print(downsampled_points.shape)
    with open(os.path.join(save_dir, 'downsampled_points_cam3inartag_new.npy'), 'wb') as f:
        np.save(f, downsampled_points)
    f.close()

    p3 = pcl.PointCloud(cam3_points_in_artag)
    downsample_points_filter = p3.make_voxel_grid_filter()
    downsample_points_filter.set_leaf_size(0.01, 0.01, 0.01)
    pts = downsample_points_filter.filter()
    downsampled_points = np.asarray(pts)
    print(downsampled_points.shape)
    with open(os.path.join(save_dir, 'downsampled_points_cam4inartag_new.npy'), 'wb') as f:
        np.save(f, downsampled_points)
    f.close()

    p4 = pcl.PointCloud(cam4_points_in_artag)
    downsample_points_filter = p4.make_voxel_grid_filter()
    downsample_points_filter.set_leaf_size(0.01, 0.01, 0.01)
    pts = downsample_points_filter.filter()
    downsampled_points = np.asarray(pts)
    print(downsampled_points.shape)
    with open(os.path.join(save_dir, 'downsampled_points_cam5inartag_new.npy'), 'wb') as f:
        np.save(f, downsampled_points)
    f.close()

    p5 = pcl.PointCloud(cam5_points_in_artag)
    downsample_points_filter = p5.make_voxel_grid_filter()
    downsample_points_filter.set_leaf_size(0.01, 0.01, 0.01)
    pts = downsample_points_filter.filter()
    downsampled_points = np.asarray(pts)
    print(downsampled_points.shape)
    with open(os.path.join(save_dir, 'downsampled_points_cam6inartag_new.npy'), 'wb') as f:
        np.save(f, downsampled_points)
    f.close()

    # cam0_points_in_artag = np.dot(cam0_to_artag, points_from_camera['cam_0'].T).T
    # ax.scatter(cam0_points_in_artag[:, 0], cam0_points_in_artag[:, 1], cam0_points_in_artag[:, 2], s=0.1)
    # from IPython import embed; embed()
    # ax.scatter(downsampled_points[:,0], downsampled_points[:,1], downsampled_points[:,2], s=0.1)
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
#     cam1_points_in_artag = np.dot(cam1_to_artag, points_from_camera['cam_1'].T).T
#     ax.scatter(cam1_points_in_artag[:, 0], cam1_points_in_artag[:, 1], cam1_points_in_artag[:, 2])
#     cam2_points_in_artag = np.dot(cam2_to_artag, points_from_camera['cam_2'].T).T
#     ax.scatter(cam2_points_in_artag[:, 0], cam2_points_in_artag[:, 1], cam2_points_in_artag[:, 2])
#     cam3_points_in_artag = np.dot(cam3_to_artag, points_from_camera['cam_3'].T).T
#     ax.scatter(cam3_points_in_artag[:, 0], cam3_points_in_artag[:, 1], cam3_points_in_artag[:, 2])
#     cam4_points_in_artag = np.dot(cam4_to_artag, points_from_camera['cam_4'].T).T
#     ax.scatter(cam4_points_in_artag[:, 0], cam4_points_in_artag[:, 1], cam4_points_in_artag[:, 2])
#     cam5_points_in_artag = np.dot(cam5_to_artag, points_from_camera['cam_5'].T).T
#     ax.scatter(cam5_points_in_artag[:, 0], cam5_points_in_artag[:, 1], cam5_points_in_artag[:, 2])
    plt.show()


if __name__ == '__main__':
    main()
