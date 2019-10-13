import os
import pickle
import register_cam1_T_camX
import numpy as np
import open3d as o3d

def subtract(dict_a, dict_b):
	pts_a = dict_a['points']
	pts_b = dict_b['points']

	presumably_object_pts = pts_a - pts_b
	# actually I will form pcd with points which are not zero
	norm_new_pts = np.linalg.norm(presumably_object_pts, axis=1)
	chosen_pts = pts_a[presumably_object_pts[:, 2] > 0.009]
	chosen_colors = dict_a['colors'][presumably_object_pts[:, 2] > 0.009]

	new_pts = np.c_[chosen_pts, chosen_colors]
	clipped_pts = register_cam1_T_camX.get_inlier_pts(new_pts,
		clip_radius=0.5)

	assert clipped_pts.shape[1] == 6, "no color in the points, not acceptable"

	new_pcd = o3d.geometry.PointCloud()
	new_pcd.points = o3d.utility.Vector3dVector(clipped_pts[:, :3])
	new_pcd.colors = o3d.utility.Vector3dVector(clipped_pts[:, 3:])
	return new_pcd

if __name__ == '__main__':
	base_pcd_a_dir = 'data/artag_only_TableDome_y2019_m10_h17_m52_s56'  # object data
	base_pcd_b_dir = 'data/artag_only_TableDome_y2019_m10_h16_m58_s11'  # empty table

	pts_and_color_dir_a = f'{base_pcd_a_dir}/pcds_in_ar_tag_frame'
	pts_and_color_dir_b = f'{base_pcd_b_dir}/pcds_in_ar_tag_frame'

	if not os.path.exists(pts_and_color_dir_a):
		raise FileNotFoundError('no object pcds in specified directory')

	if not os.path.exists(pts_and_color_dir_b):
		raise FileNotFoundError('no empty pcds in specified directory')

	all_a_files = [os.path.join(pts_and_color_dir_a, file) for file in os.listdir(pts_and_color_dir_a)]
	all_b_files = [os.path.join(pts_and_color_dir_b, file) for file in os.listdir(pts_and_color_dir_b)]

	only_object_pcds = list()
	for a, b in zip(all_a_files, all_b_files):
		with open(a, 'rb') as f:
			dict_a = pickle.load(f)
		f.close()

		with open(b, 'rb') as f:
			dict_b = pickle.load(f)
		f.close()

		new_pcd = subtract(dict_a, dict_b)
		only_object_pcds.append(new_pcd)

	o3d.visualization.draw_geometries([only_object_pcds[1], only_object_pcds[4], only_object_pcds[5]])

	care_about_cams = [1, 4, 5]  # Need to figure out what is wrong with camera 0, 2, 3 
	# now I will tightly align these three point clouds using coloredICP
	cam1_T_cam4 = register_cam1_T_camX.run_color_icp(only_object_pcds[5], only_object_pcds[1])
