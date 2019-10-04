import os
import gin
import copy
import numpy as np
import pickle
import open3d as o3d
import utils

def make_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def visualize(list_of_pcds):
    o3d.visualization.draw_geometries(list_of_pcds)


def draw_registration_results(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0]) # something closer to orange
    target_temp.paint_uniform_color([0, 0.651, 0.929]) # something closer to blue
    source_temp.transform(transformation)
    visualize([source_temp, target_temp])


def get_inlier_pts(pts, clip_radius=5.0):
    """
    Assumptions points are centered at (0,0,0)
    only includes the points which falls inside the desired radius
    :param pts: a numpy.ndarray of form (N, 3)
    :param clip_radius: a float
    :return: numpy.ndarray of form (Ninliers, 3)
    """
    sq_radius = clip_radius ** 2
    pts_norm_squared = np.linalg.norm(pts, axis=1)
    idxs = (pts_norm_squared - sq_radius) <= 0.0
    chosen_pts = pts[idxs]
    return chosen_pts


def prepare_data(depths, intrinsics, center=True, clip_radius=None):
    """
    forms the pcd by unprojecting depth image and open3d
    :param depths: all the depth images
    :param intrinsics: all the intrinsics
    :return: returns pcds for all the cameras
    """
    num_images = len(depths)
    pcds = list()
    for i in range(num_images):
        pts = utils._unproject_using_depth(depths[i], intrinsics[i], depth_scale=1000.0)
        if center:
            pts_mean = pts.mean(axis=0)
            pts -= pts_mean
        if clip_radius:
            pts = get_inlier_pts(pts, clip_radius=clip_radius)
        pcd = make_pcd(pts)
        pcds.append(pcd)
    return pcds


def compute_features(pcd, radius_features):
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_features, max_nn=100)
    )
    return pcd_fpfh


def execute_fast_global_registration(source, target,
                                     source_fpfh, target_fpfh,
                                     voxel_size):
    distance_threshold = voxel_size
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )
    )
    return result


def runICP(source, target, transform_init, threshold, mode='point2plane'):
    refined_reg = o3d.registration.registration_icp(
        source, target, threshold, transform_init,
        o3d.registration.TransformationEstimationPointToPlane() if mode=='point2plane' else\
        o3d.registration.TransformationEstimationPointToPoint()
    )
    return refined_reg


def align(source, target, trans_init=np.eye(4), mode='point2plane',
          radius_KDTree=0.1, nearest_neighbors_for_plane=30, voxel_size=0.05,
          segment_plane=True):
    """
    transforms point cloud using different methods
    :param source: source pointcloud
    :param target: target_pointcloud
    :param threshold: a float value determining the threshold
    :param trans_init: initial transformation
    :param mode: "global" runs FGR to initialize trans_init and then runs point2plane ICP
                 "point2point" runs point to point ICP for the pointsets (this should almost never be used)
                 "point2plane" runs point to plane ICP for the pointsets
    :param radius_KDTree: radius of search for KDTree to compute vertex normals (estimate the plane)
    :param nearest_neighbors_for_plane: max number of nearest neighbors to consider for fitting the plane
    :param voxel_size: a float giving the voxel size
    :param segment_plane: do you want to segment plane from the pointcloud
    :return: transformation of the aligned pointcloud
    """
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_KDTree, max_nn=nearest_neighbors_for_plane
    ))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_KDTree, max_nn=nearest_neighbors_for_plane
    ))

    if segment_plane:
        # now I want to segment out the plane
        new_source = np.asarray(source.points)
        new_target = np.asarray(target.points)

        # next compute the covariance matrix
        cov_source = np.dot(new_source.T, new_source)
        cov_target = np.dot(new_target.T, new_target)

        # do the eigen decompostions
        source_w, source_v = np.linalg.eigh(cov_source)
        target_w, target_v = np.linalg.eigh(cov_target)

        # here I do have the equation of the normal, I just have to find one point which lies on the plane
        normal_source = source_v[:, 0]
        normal_target = target_v[:, 0]

        # source plane and target plane
        source_x = new_source[0, :]
        source_vectors = new_source[1:, :] - source_x
        dots = np.dot(source_vectors, normal_source)
        # TODO: the next line is specific for the dataset
        chosen_idx_source = dots <= 0.
        chosen_idx_source = np.concatenate([[False], chosen_idx_source])
        chosen_source_points = new_source[chosen_idx_source]

        target_x = new_target[0, :]
        target_vectors = new_target[1:, :] - target_x
        dots = np.dot(target_vectors, normal_target)
        # TODO: again the next line is specific to the dataset
        chosen_idx_targets = dots <= 0.
        chosen_idx_targets = np.concatenate([[False], chosen_idx_targets])
        chosen_target_points = new_target[chosen_idx_targets]

        new_source_pcd = o3d.geometry.PointCloud()
        new_source_pcd.points = o3d.utility.Vector3dVector(chosen_source_points)

        new_target_pcd = o3d.geometry.PointCloud()
        new_target_pcd.points = o3d.utility.Vector3dVector(chosen_target_points)

        visualize([new_source_pcd, new_target_pcd])


    if mode == "global":
        # run the global registration first
        # compute the features first
        source_fpfh = compute_features(source, 5*voxel_size)
        target_fpfh = compute_features(target, 5*voxel_size)
        fast_registration_result = execute_fast_global_registration(
            source, target, source_fpfh, target_fpfh,
            0.5* voxel_size
        )

        draw_registration_results(source, target, fast_registration_result.transformation)
        # now using this result as initialization, I will do the ICP plane to plane
        refined_result = runICP(source, target, fast_registration_result.transformation, voxel_size*0.4)
        return refined_result

    if mode == 'point2plane':
        reg_result = runICP(source, target, trans_init, voxel_size*0.4, mode=mode)
        return reg_result

    if mode == 'point2point':
        reg_result = runICP(source, target, trans_init, voxel_size*0.4, mode=mode)
        return reg_result


@gin.configurable
def main(data_file_path, trans_init=np.eye(4),
         mode='global', radius_KDTree=0.1, nearest_neighbors_for_plane=30,
         voxel_size=0.05, segment_plane=False):

    # main is just for loading the data, this function should return form an array of imgs, depths, exts and ints
    if not os.path.exists(data_file_path):
        raise FileNotFoundError('did not find the data file please check the path')

    # now form the array as mentioned in the comment above
    with open(data_file_path, 'rb') as f:
        raw_data = pickle.load(f, encoding='latin1')

    # a hack for the data and for loop to run properly
    raw_data['cam_0_to_0'] = np.eye(4)

    # the next line is the problem of the data, can be easily rectified
    interesting_cams = [0, 1, 3, 4, 5]
    rgbs, depths, ints, exts = list(), list(), list(), list()
    for i in interesting_cams:
        rgb_img = raw_data[f'img{i+1}']
        depth_img = raw_data[f'depth_img{i+1}']
        img_int = raw_data[f'cam_{i+1}_intrinsics']
        img_ext = raw_data[f'cam_0_to_{i}']
        rgbs.append(rgb_img), depths.append(depth_img)
        ints.append(img_int), exts.append(img_ext)

    # stack all of them to form the array
    rgbs, depths = np.stack(rgbs), np.stack(depths)
    ints, exts = np.stack(ints), np.stack(exts)

    # form all the pcds, actually I only need the depths for now
    # TODO: make the colored pointcloud
    pcds = prepare_data(depths, ints, center=True, clip_radius=0.4)

    # do the registration
    source_pcd, target_pcd = pcds[2], pcds[3]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.5)
    visualize([source_pcd, target_pcd, frame])
    reg_result = align(source_pcd, target_pcd, trans_init, mode, radius_KDTree, nearest_neighbors_for_plane, voxel_size,
                       segment_plane=segment_plane)
    draw_registration_results(source_pcd, target_pcd, reg_result.transformation)


if __name__ == '__main__':
    gin_config_file_path = "data/gin_config_file.gin"
    # parse the config file
    gin.parse_config_file(gin_config_file_path)
    main()