import os
import pickle
import register_cam1_T_camX
import numpy as np
import open3d as o3d

def subtract(dict_a, dict_b, vis=True):
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
	if vis:
		o3d.visualization.draw_geometries([new_pcd])
	return new_pcd


def get_alignment_to_cam1(pcds):
	"""
		pcds: list of pcds for which to compute the transformation
		returns: dict with keys cam1_T_camX
	"""
	transformations = dict()
	t = np.eye(4)
	transformations['cam1_T_cam1'] = t
	for i in range(1, len(pcds)-1):
		target = pcds[i]
		source = pcds[i+1]

		### ... Trying the first one ... ###
		camPrev_T_camNext = register_cam1_T_camX.run_color_icp(source, target, vis=False)
		t = np.dot(t, camPrev_T_camNext)
		transformations[f'cam1_T_cam{i+1}'] = t
	return transformations


def transform_all_to_cam1(pcds, transformations):
	"""Transforms all pcds to camera1 using the transformations computed using colored ICP
	prameters:
	------------
		pcds: all_object_pcds files
		transformations: dict containing the transforms
	returns:
	------------
		list: all_pcds_in_cam1_frame
	"""
	all_pcds_in_cam1 = []
	for i in range(1, len(pcds)):
		temp_pts = np.asarray(pcds[i].points)
		temp_colors = np.asarray(pcds[i].colors)

		# transform using the matrix
		temp_pts = np.c_[temp_pts, np.ones(len(temp_pts))]
		new_pts = np.dot(transformations[f'cam1_T_cam{i}'], temp_pts.T).T
		new_pts = np.c_[new_pts[:, :3], temp_colors]

		new_pcd = register_cam1_T_camX.make_pcd(new_pts)
		all_pcds_in_cam1.append(new_pcd)
	return all_pcds_in_cam1


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

		new_pcd = subtract(dict_a, dict_b, vis=False)
		only_object_pcds.append(new_pcd)

	o3d.visualization.draw_geometries(only_object_pcds[1:])

	# now i will get tight alignment with camera1 of every other camera
	transformations = get_alignment_to_cam1(only_object_pcds)
	assert len(transformations.keys()) == 5, "some of the transformations are missing"

	# I now have all the relative transformations, need to apply them and see the scene
	all_pcds_in_cam1 = transform_all_to_cam1(only_object_pcds, transformations)
	assert len(all_pcds_in_cam1) == 5, "some of the pcds are missing"

	o3d.visualization.draw_geometries(all_pcds_in_cam1)

	import ipdb; ipdb.set_trace()