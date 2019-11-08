# import os
# import gin
# import copy
# import numpy as np
# import pickle
import open3d as o3d
import utils_vis
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
# from IPython import embed
# from  os.path import join
import ipdb
st = ipdb.set_trace
import sys
from  os.path import join
import os

def make_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # if the dim is greater than 3 I expect the color
    if pts.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
            if pts[:, 3:].max() > 1. else pts[:, 3:])
    return pcd

def merge_pcds(pcds):
    pts = [np.asarray(pcd.points) for pcd in pcds]
    colors = [np.asarray(pcd.colors) for pcd in pcds]
    # assert len(pts) == 5, "these is the number of supplied pcd, it should match"
    combined_pts = np.concatenate(pts, axis=0)
    combined_colors = np.concatenate(colors, axis=0)
    assert combined_pts.shape[1] == 3, "concatenation is wrong"
    return combined_pts, combined_colors



def visualize(list_of_pcds):
    o3d.visualization.draw_geometries(list_of_pcds)


# def draw_registration_results(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0]) # something closer to orange
#     target_temp.paint_uniform_color([0, 0.651, 0.929]) # something closer to blue
#     source_temp.transform(transformation)
#     visualize([source_temp, target_temp])


def get_inlier_pts(pts, clip_radius=5.0):
    """
    Assumptions points are centered at (0,0,0)
    only includes the points which falls inside the desired radius
    :param pts: a numpy.ndarray of form (N, 3)
    :param clip_radius: a float
    :return: numpy.ndarray of form (Ninliers, 3)
    """
    # do the mean centering of the pts first, deepcopy for this
    filter_pts = copy.deepcopy(pts[:, :3])
    mean_pts = filter_pts.mean(axis=0)
    assert mean_pts.shape == (3,), "wrong mean computation"
    filter_pts -= mean_pts

    sq_radius = clip_radius ** 2
    pts_norm_squared = np.linalg.norm(filter_pts, axis=1)
    idxs = (pts_norm_squared - sq_radius) <= 0.0
    chosen_pts = pts[idxs]
    return chosen_pts


def prepare_data(depths, intrinsics, rgbs, vis=False,
    save_data="images"):
    """
    forms the pcd by unprojecting depth image and open3d
    :param depths: all the depth images
    :param intrinsics: all the intrinsics
    :param rgbs: all the color images
    :param center: if True I subtract the mean point from the unproject points
    :parm clip_radius: if specified I only take points which lie with the sphere
    :param vis: if specified I visualize all the intermediate pointclouds formed
    :return: returns pcds for all the cameras
    """
    if save_data:
        if not os.path.exists(save_data):
            os.makedirs(save_data)
    num_images = len(depths)
    pcds = list()
    for i in range(num_images):
        if save_data:
            plt.imshow(np.clip(depths[i], 0, 600))
            plt.imshow(rgbs[i])
        pts = utils_vis.vectorized_unproject_using_depth(depths[i], intrinsics[i],
            rgbs[i], depth_scale=1000.0)
        # if center:
        #     pts_mean = pts.mean(axis=0)
        #     pts -= pts_mean
        # if clip_radius is not None:
        #     pts = get_inlier_pts(pts, clip_radius=clip_radius)
        pcd = make_pcd(pts)
        # if vis:
        #     # make a coordinate frame
        #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0., 0., 0.])
        #     visualize([pcd, frame])
        pcds.append(pcd)
    return pcds


# def compute_features(pcd, radius_features):
#     pcd_fpfh = o3d.registration.compute_fpfh_feature(
#         pcd,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_features, max_nn=100)
#     )
#     return pcd_fpfh


# def execute_fast_global_registration(source, target,
#                                      source_fpfh, target_fpfh,
#                                      voxel_size):
#     distance_threshold = voxel_size
#     result = o3d.registration.registration_fast_based_on_feature_matching(
#         source, target, source_fpfh, target_fpfh,
#         o3d.registration.FastGlobalRegistrationOption(
#             maximum_correspondence_distance=distance_threshold
#         )
#     )
#     return result


# def runICP(source, target, transform_init, threshold, mode='point2plane'):
#     refined_reg = o3d.registration.registration_icp(
#         source, target, threshold, transform_init,
#         o3d.registration.TransformationEstimationPointToPlane() if mode=='point2plane' else\
#         o3d.registration.TransformationEstimationPointToPoint()
#     )
#     return refined_reg


# def align(source, target, trans_init=np.eye(4), mode='point2plane',
#           radius_KDTree=0.1, nearest_neighbors_for_plane=30, voxel_size=0.05):
#     """
#     transforms point cloud using different methods
#     :param source: source pointcloud
#     :param target: target_pointcloud
#     :param threshold: a float value determining the threshold
#     :param trans_init: initial transformation
#     :param mode: "global" runs FGR to initialize trans_init and then runs point2plane ICP
#                  "point2point" runs point to point ICP for the pointsets (this should almost never be used)
#                  "point2plane" runs point to plane ICP for the pointsets
#     :param radius_KDTree: radius of search for KDTree to compute vertex normals (estimate the plane)
#     :param nearest_neighbors_for_plane: max number of nearest neighbors to consider for fitting the plane
#     :param voxel_size: a float giving the voxel size
#     :param segment_plane: do you want to segment plane from the pointcloud
#     :return: transformation of the aligned pointcloud
#     """
#     source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
#         radius=radius_KDTree, max_nn=nearest_neighbors_for_plane
#     ))
#     target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
#         radius=radius_KDTree, max_nn=nearest_neighbors_for_plane
#     ))
    
#     if mode == "global":
#         # run the global registration first
#         # compute the features first
#         source_fpfh = compute_features(source, 5*voxel_size)
#         target_fpfh = compute_features(target, 5*voxel_size)
#         fast_registration_result = execute_fast_global_registration(
#             source, target, source_fpfh, target_fpfh,
#             0.5* voxel_size
#         )

#         draw_registration_results(source, target, fast_registration_result.transformation)
#         # now using this result as initialization, I will do the ICP plane to plane
#         refined_result = runICP(source, target, fast_registration_result.transformation, voxel_size*0.4)
#         return refined_result

#     if mode == 'point2plane':
#         reg_result = runICP(source, target, trans_init, voxel_size*0.4, mode=mode)
#         return reg_result

#     if mode == 'point2point':
#         reg_result = runICP(source, target, trans_init, voxel_size*0.4, mode=mode)
#         return reg_result


def ar_tag_data_loader(ar_tag_data_path,vis=True):
    # first I will load the depths and the color images from each camera
    rgbs_depths_base_path = f"{ar_tag_data_path}/rgb_depth_npy"
    rgb_data_path = f"{rgbs_depths_base_path}/colorData"
    depth_data_path= f"{rgbs_depths_base_path}/depthData"

    rgbs, depths, ints, exts = list(), list(), list(), list()

    # now I need to load the intrinsics and the extrinsics
    # NOTE: The extrinsics are from ar_tag_T_camX form need to covert them to CamX_T_camY form
    registration_path = f"{ar_tag_data_path}/ar_tag"
    ints_path = f"{registration_path}/intrinsics.pkl"
    exts_path = f"{registration_path}/extrinsics.npy"
    exts_data = np.load(exts_path)

    with open(ints_path, 'rb') as f:
        int_data = pickle.load(f, encoding='latin1')

    # now I take one image from each of the camera, it should be the same timestep Image
    num_cams = len(os.listdir(depth_data_path))
    for i in range(1, num_cams+1):
        rgb_img_path = f"{rgb_data_path}/Cam{i}/cam_{i}_color_9.npy"
        rgb_img = np.load(rgb_img_path)
        rgbs.append(rgb_img)

        if vis:
            plt.imshow(rgb_img)
            plt.show()

        # load the depth image
        depth_img_path = f"{depth_data_path}/Cam{i}/cam_{i}_depth_9.npy"
        depth_img = np.load(depth_img_path)
        depths.append(depth_img)

        if vis:
            plt.imshow(np.clip(depth_img, -0.5, 0.5))
            plt.show()

        # load up the intrinsics
        ints.append(int_data[f'{i}'])

        # load up the extrinsics for this camera

        exts.append(np.linalg.inv(exts_data[i-1]))

    # stack'em up and return :)
    rgbs, depths = np.stack(rgbs), np.stack(depths)
    ints, exts = np.stack(ints), np.stack(exts)

    return rgbs, depths, ints, exts


# def draw_registration_result_original_color(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target])


# @gin.configurable
# def run_color_icp(source, target, voxel_radius=[0.04, 0.02, 0.01],
#     max_iter=[50, 30, 14], vis=False):
#     print(voxel_radius)
#     print(max_iter)
#     current_transformation = np.identity(4)
#     print("1. Colored point cloud registration")
#     for scale in range(3):
#         iter = max_iter[scale]
#         radius = voxel_radius[scale]
#         print([iter, radius, scale])

#         print("1-1. Downsample with a voxel size %.2f" % radius)
#         source_down = source.voxel_down_sample(radius)
#         target_down = target.voxel_down_sample(radius)

#         print("1-2. Estimate normal.")
#         source_down.estimate_normals(
#             o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
#         target_down.estimate_normals(
#             o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

#         print("3-3. Applying colored point cloud registration")
#         result_icp = o3d.registration.registration_colored_icp(
#             source_down, target_down, radius, current_transformation,
#             o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
#                                                     relative_rmse=1e-6,
#                                                     max_iteration=iter))
#         current_transformation = result_icp.transformation
#         print(result_icp)
#     if vis:
#         draw_registration_result_original_color(source, target,
#                                             result_icp.transformation)
#     return result_icp.transformation

# rans_init=np.eye(4),
#          mode='global', radius_KDTree=0.1, nearest_neighbors_for_plane=30,
#          voxel_size=0.05, clip_radius=1., ar_tag_data_path=None,
#          use_ar_tag=False, vis=False
def main(ar_tag_data_path):

    ### ... Loading ar_tag data ... ###
    # check if the ar_tag data path is provided
    if not os.path.exists(ar_tag_data_path):
        raise FileNotFoundError('should provide a valid directory containing ar_tag data')


    # load the ar_tag data
    rgbs, depths, ints, exts = ar_tag_data_loader(ar_tag_data_path,vis=False)

    # form all the pcds, actually I only need the depths for now
    pcds = prepare_data(depths, ints, rgbs)

    print('view all pcds in camera frame')
    if False:
        visualize(pcds)

    # you have the ar_tag extrinsics and the pcds just rotate all the points and view
    recon_pcds = list()
    for i in range(len(pcds)):
        temp_pts = np.asarray(pcds[i].points)
        temp_colors = copy.deepcopy(np.asarray(pcds[i].colors))

        # multiply with the corresponding extrinsics to transform to world space
        temp_pts = np.c_[temp_pts, np.ones(len(temp_pts))]
        new_pts = np.dot(np.linalg.inv(exts[i]), temp_pts.T).T

        # now you have the points in world frame, form the colored pointcloud here
        new_pts = np.c_[new_pts[:, :3], temp_colors]

        new_pcd = make_pcd(new_pts)
        recon_pcds.append(new_pcd)

    # st()
    # visualize the pcds and move on
    print('view all pcds in ar_tag frame')
    # for i in range(len(recon_pcds)):
    combined_pts, combined_colors = merge_pcds(recon_pcds)
    combined_pts_mean = combined_pts.mean(axis=0)
    combined_pts -= combined_pts_mean
    combined_points = np.concatenate([combined_pts, combined_colors],axis=1) 
    combined_points = get_inlier_pts(combined_points, clip_radius=1.0)
    combined_pcds = make_pcd(combined_points)
    visualize([combined_pcds])
    st()    

    # Here we have data from all the images in common ar_tag frame, save it
    # create a new directory in the ar_tag_data_path, pcds_in_ar_tag_frame
    # recon_pcds_save_path = f"{ar_tag_data_path}/pcds_in_ar_tag_frame"
    # if not os.path.exists(recon_pcds_save_path):
    #     os.makedirs(recon_pcds_save_path)

    # # save all the pcds
    # for i in range(len(recon_pcds)):
    #     save_file_path = f'{recon_pcds_save_path}/pcd_{i}.pkl'
    #     with open(save_file_path, 'wb') as f:
    #         points = np.asarray(recon_pcds[i].points)
    #         colors = np.asarray(recon_pcds[i].colors)
    #         save_dict = {
    #             'points': points,
    #             'colors': colors
    #         }
    #         pickle.dump(save_dict, f)
    #     f.close()


if __name__ == '__main__':
    root_dir = "/home/zhouxian/data/TableDome"
    mydir = sys.argv[1]
    folder = join(root_dir,mydir)
    main(folder)
