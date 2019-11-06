import open3d as o3d
import numpy as np
import yaml
import os

def get_intrinsics_from_yaml(yaml_fname, cam_name):
  with open(yaml_fname, "r") as file_handle:
      calib_data = yaml.load(file_handle)

  all_cam_info = calib_data[cam_name]
  cam_intrinsics = all_cam_info['intrinsics']
  intrinsics_mat = np.eye(3)
  intrinsics_mat[0][0] = cam_intrinsics[0]
  intrinsics_mat[1][1] = cam_intrinsics[1]
  intrinsics_mat[0][2] = cam_intrinsics[2]
  intrinsics_mat[1][2] = cam_intrinsics[3]
  return intrinsics_mat

def parse_intrinsics(intrinsics):
  return intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]

def vectorized_unproject(depth, intrinsics, rgb=None, depth_scale=1., depth_trunc=1000.):
  fx, fy, cx, cy = parse_intrinsics(intrinsics)
  print(fx, fy, cx, cy)
  # first scale the entire depth image
  depth /= depth_scale

  # form the mesh grid
  xv, yv = np.meshgrid(np.arange(depth.shape[1], dtype=float), np.arange(depth.shape[0], dtype=float))

  xv -= cx
  xv /= fx
  xv *= depth
  yv -= cy
  yv /= fy
  yv *= depth
  points = np.c_[xv.flatten(), yv.flatten(), depth.flatten()]

  if rgb is not None:
      # flatten it and add to the points
      rgb = rgb.reshape(-1, 3)

  if rgb is not None:
    points = np.concatenate((points, rgb), axis=1)
  return points


def get_depth_images(filename):
  if not os.path.exists(filename):
    raise FileNotFoundError

  depths = np.load(filename)
  depth_m = depths / 1000.
  return depth_m

def visualize(unprojected_points, rgbs=None):
  pcds = list()
  for i in range(len(unprojected_points)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unprojected_points[i])
    if rgbs is not None:
      curr_rgb = rgbs[i].reshape(-1, 3)
      pcd.colors = o3d.utility.Vector3dVector(curr_rgb/255.)
    else:
      if i == 0:
        pcd.paint_uniform_color([1, 0, 0])
      if i == 1:
        pcd.paint_uniform_color([0, 0, 1])
    pcds.append(pcd)

  o3d.visualization.draw_geometries(pcds)

def make_scene():
  cam_names_for_yaml = ['cam2', 'cam3']
  depth_file_path = 'depths_1105.npy'
  color_file_path = 'colors_1105.npy'
  yaml_file_path = '/home/zhouxian/catkin_ws/src/calibrate/src/scripts/camchain-homezhouxiandataTableDomecam_calibrateTableDome_y2019_m10_h18_m54_s18Checkerboard_CalibData.yaml'
  extrinsics_file_path = 'cam4_T_cam3.npy'

  # intrinsics = list()
  # for cam_name in cam_names_for_yaml:
  #   intrinsics.append(get_intrinsics_from_yaml(yaml_file_path, cam_name))

  # intrinsics = np.stack(intrinsics)

  # # the hard coded are the true intrinsics of intel
  intrinsics_cam3 = np.eye(3)
  intrinsics_cam3[0][0] = 613.620849609375
  intrinsics_cam3[1][1] = 613.7835083007812
  intrinsics_cam3[0][2] = 325.1519775390625
  intrinsics_cam3[1][2] = 243.37289428710938
  
  intrinsics_cam4 = np.eye(3)
  intrinsics_cam4[0][0] = 612.0875244140625
  intrinsics_cam4[1][1] = 612.2713623046875
  intrinsics_cam4[0][2] = 322.45697021484375
  intrinsics_cam4[1][2] = 244.1866455078125

  intrinsics = np.stack([intrinsics_cam3, intrinsics_cam4])

  # get the extrinsics now
  extrinsics = np.load('cam4_T_cam3.npy')

  # get the depths
  depths = get_depth_images(depth_file_path)
  rgbs = np.load(color_file_path)

  # make the scene
  unprojected_points = list()
  for i in range(len(cam_names_for_yaml)):
    pts = vectorized_unproject(depths[i], intrinsics[i])
    unprojected_points.append(pts)

  unprojected_points = np.stack(unprojected_points)
  
  # visualize
  visualize(unprojected_points, rgbs=rgbs)

  # run the extrinsics and be done with it
  cam3_pts = unprojected_points[0]
  cam4_pts = unprojected_points[1]

  cam3_homogenous = np.c_[cam3_pts[:, 0], cam3_pts[:, 1], cam3_pts[:, 2], np.ones(len(cam3_pts))]
  cam4_T_cam3_pts = np.dot(extrinsics, cam3_homogenous.T).T[:, :3]
  
  # visualize [red, blue] predicted
  visualize([cam4_T_cam3_pts, cam4_pts], rgbs=rgbs)

if __name__ == '__main__':
  make_scene()