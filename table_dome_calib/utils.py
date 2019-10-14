import os
import copy
import pathlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import transformations
from pathlib import Path
from scipy.linalg import inv, sqrtm
import PIL.Image as Image

import pyglet
pyglet.options['shadow_window'] = False
import trimesh
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     Primitive, Mesh, Node, Scene, Viewer,\
                     OffscreenRenderer


def visualize(points_list, origin, clip_radius=2.):
    # convert all points to pcds and visualize using open3d
    pcds = list()
    for points in points_list:
        inliers = get_inliers(points, clip_radius)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inliers)
        pcds.append(pcd)

    # create mesh frame
    frame = o3d.geometry.create_mesh_coordinate_frame(origin=origin, size=2.)
    pcds.append(frame)
    o3d.visualization.draw_geometries(
        pcds
    )


def get_inliers(points, clip_radius):
    sqrt_radius = clip_radius ** 0.5
    points_norm = np.linalg.norm(points, axis=1)
    check = points_norm - sqrt_radius
    idxs = check <= 0.0
    inlier_pts = points[idxs, :]
    return inlier_pts


# vectorized version of unprojecting using depth image
def vectorized_unproject_using_depth(depth, intrinsics, rgb, depth_scale):
    depth = depth.astype(np.float32)
    fx, fy, cx, cy = parse_intrinsics(intrinsics)
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

    points = np.concatenate((points, rgb), axis=1)
    return points


def _unproject_using_depth(depth_img, intrinsics_mat, depth_scale=1.0):
    points = list()
    fx, fy, cx, cy = parse_intrinsics(intrinsics_mat)
    # NOTE: the image is height, width so x->j and y->i
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            pz = depth_img[i][j] / depth_scale
            px = (j - cx) * pz / fx
            py = (i - cy) * pz / fy
            points.append(np.asarray([px, py, pz]))
    points = np.stack(points)
    assert points.shape[1] == 3, "stacking is wrong"
    return points


def dm_unproject_using_depth(depth_img, intrinsics_mat):
    points = list()
    fx, fy, cx, cy = parse_intrinsics(intrinsics_mat)
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            pz = depth_img[depth_img.shape[0]-1-i][j]
            px = (j-cx) * pz / fx
            py = (depth_img.shape[0]-1-i -cy) * pz / fy
            points.append(np.asarray([px, py, pz]))
    points = np.stack(points)
    return points


def parse_intrinsics(intrinsics_mat):
    fx = intrinsics_mat[0][0]
    fy = intrinsics_mat[1][1]
    cx = intrinsics_mat[0][2]
    cy = intrinsics_mat[1][2]
    return fx, fy, cx, cy


def recreate_scene(depths, intrinsics, extrinsics,
    camR_T_origin, clip_radius=2.0, mujoco_py_scene=True):
    """
    parameters:
    ----------------
        depths : all depth images
        intrinsics : all intrinsics to unproject in cam frame
        extrinsics : all extrinsics to bring to world frame
            NOTE: extrinsics are like origin_T_camX
        camR_T_origin : takes from origin to camR coordinates
        clip_radius : for better visualization of point cloud
        TODO: use open3d to do the same thing
    """
    camX_xyzs = list()
    world_xyzs = list()
    camR_xyzs = list()
    visualize_camXs = True
    visualize_world = True
    visualize_camRs = True
    for i in range(len(depths)):
        camX_xyz = _unproject_using_depth(depths[i], intrinsics[i]) if mujoco_py_scene\
            else dm_unproject_using_depth(depths[i], intrinsics[i])
        # project this into image
        camX_xyzs.append(camX_xyz)

        # now bring it to world coordinate
        world_xyz_homogenous = np.c_[camX_xyz, np.ones(len(camX_xyz))]
        world_xyz = np.dot(extrinsics[i], world_xyz_homogenous.T).T
        world_xyzs.append(world_xyz[:,:3])

        # finally bring it back to ref_cam frame, here world_xyz
        # are in homogenous system
        camR_xyz = np.dot(camR_T_origin, world_xyz.T).T
        camR_xyzs.append(camR_xyz[:, :3])

    if visualize_camXs:
        visualize(camX_xyzs, origin=np.zeros(3), clip_radius=clip_radius)

    if visualize_world:
        visualize(world_xyzs, origin=np.zeros(3), clip_radius=clip_radius)

    if visualize_camRs:
        # when visualized in camR world origin is far from the scene, hence the
        # origin when kept at [1.25, 0.41, 0.45] shows far away
        visualize(camR_xyzs, origin=np.asarray([0., 0., 0]),
            clip_radius=clip_radius)


def _convert_depth_to_meters(sim, depth):
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image


def get_intrinsics(sim, cam_no, img_width, img_height):
    """
        to compute fovx check this out
        https://stackoverflow.com/questions/5504635/computing-fovx-opengl
    """
    fovy = sim.model.cam_fovy[cam_no]
    # now compute the fovx
    aspect = float(img_width) / img_height
    assert aspect == 1., "I am giving data such that aspect is 1"
    fovx = 2 * np.arctan(np.tan(np.deg2rad(fovy) * 0.5) * aspect)
    fovx = np.rad2deg(fovx)
    cx = img_width / 2.
    cy = img_height / 2.
    fx = cx / np.tan(np.deg2rad(fovx / 2.))
    fy = cy / np.tan(np.deg2rad(fovy / 2.))
    K = np.zeros((3,3), dtype=np.float)
    K[2][2] = 1
    K[0][0] = fx
    K[1][1] = fy
    K[0][2] = cx
    K[1][2] = cy
    return K


def dm_get_intrinsics(physics, cam_no, img_width, img_height):
    """
        physics:    dm_control physics simulator object
        cam_no:     camera number in the scene
        img_width:  width of the image
        img_height: height of the image
    """
    fovy = physics.model.cam_fovy[cam_no]
    # now compute the fovx
    aspect = float(img_width) / img_height
    assert aspect == 1., "I am giving data such that aspect is 1"
    fovx = 2 * np.arctan(np.tan(np.deg2rad(fovy) * 0.5) * aspect)
    fovx = np.rad2deg(fovx)
    cx = img_width / 2.
    cy = img_height / 2.
    fx = cx / np.tan(np.deg2rad(fovx / 2.))
    fy = cy / np.tan(np.deg2rad(fovy / 2.))
    K = np.zeros((3,3), dtype=np.float)
    K[2][2] = 1
    K[0][0] = fx
    K[1][1] = fy
    K[0][2] = cx
    K[1][2] = cy
    return K


def convert_to_XYZ_format(set_of_points,
        save_dir="/Users/macbookpro15/Documents/mujoco_hand_exps/data/sensor_meshes"):
    """
    set of points: numpy.ndarray of list of points
    save_dir: where do you want to store the pts later used for mesh creation
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # for each set of points do the conversion into xyz format
    num_pts_sets = len(set_of_points)
    for pt in range(num_pts_sets):
        a_pt = set_of_points[pt]
        # subtract the center while saving these points
        a_pt = a_pt - a_pt[0]
        with open(f'{save_dir}/pts_{pt}.XYZ', 'w') as f:
            for p in a_pt:
                f.write(f'{p[0]} {p[1]} {p[2]}\n')
        f.close()
    return True


def convert_to_mesh(pts_dir, mesh_dir):
    """
    pts_dir: the directory which is used above for storing pts, or any dir
             which contains .XYZ files
    mesh_dir: the directory in which you want to store the resulting meshes
    """
    if not os.path.exists(pts_dir):
        raise FileNotFoundError('please specify a valid pts directory')

    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    meshlabserver_path = "/Applications/meshlab.app/Contents/MacOS/meshlabserver"
    script_file_path = "/Users/macbookpro15/Documents/mujoco_hand_exps/mujoco_hand_exps/another_script.mlx"
    # now go through each file in the pts dir which has .XYZ extension
    for xyz_file in os.listdir(pts_dir):
        if xyz_file.endswith('XYZ'):
            inp_file_path = f'{pts_dir}/{xyz_file}'
            # get the number out
            time_step = xyz_file[4:-4]
            print('timestep=', time_step)
            out_dir_path = f'{mesh_dir}/mesh_{time_step}'
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path)
            project_path = f"{out_dir_path}/project_{time_step}.mlp"
            command = f'{meshlabserver_path} -i {inp_file_path} -w {project_path} -s {script_file_path}'
            command = command.split(' ')
            cmd = subprocess.Popen(command)
            cmd.communicate()
    print('done')


def get_pos_mat_from_mujoco(sim, cam_name):
    pos = sim.data.get_camera_xpos(cam_name)
    mat = sim.data.get_camera_xmat(cam_name)
    rot_mat = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    mat = np.dot(mat, rot_mat)
    return pos, mat


def get_extrinsics(sim, cam_name):
    """
        TODO: for some reason this returns me the -ve z-axis
              fix that
        TODO: also the extrinsic is correct for the flipped images
              fix that too
    """
    pos = sim.data.get_camera_xpos(cam_name)
    mat = sim.data.get_camera_xmat(cam_name)
    # left_t_right = np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    # new_mat = np.dot(left_t_right, mat)

    trans = np.dot(mat, -pos)
    # form the matrix
    ext_mat = np.zeros((4, 4), dtype=float)
    ext_mat[:3, :3] = mat
    ext_mat[3][3] = 1
    ext_mat[:3, 3] = trans
    return ext_mat


def dm_get_extrinsics(physics, cam_id):
    """
        physics: dm_control physics simulator object
        cam_id : id of the camera we want extrinsics for
    """
    pos = physics.data.cam_xpos[cam_id]
    mat = physics.data.cam_xmat[cam_id].reshape(3,3)
    rot_mat = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    mat = np.dot(mat, rot_mat)
    ext = np.eye(4)
    ext[:3, :3] = mat
    ext[:3, 3] = pos
    return ext


def ref_cam_extrinsics(physics, cam_id):
    pos = physics.data.cam_xpos[cam_id]
    mat = physics.data.cam_xmat[cam_id].reshape(3, 3)

    trans = np.dot(mat, -pos)
    ext_mat = np.eye(4)
    ext_mat[:3, :3] = mat
    ext_mat[3][3] = 1
    ext_mat[:3, 3] = trans
    return ext_mat


def get_imgs(sim, cam_name, img_height, img_width,
    save_to_disk=False):
    rgb, depth = copy.deepcopy(sim.render(img_width, img_height,\
        depth=True, camera_name=cam_name))

    depth = _convert_depth_to_meters(sim, depth)
    if save_to_disk:
        log_dir = "/Users/macbookpro15/Documents/inp_imgs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        rgb_path = f"{log_dir}/rgb_{cam_name}.jpg"
        depth_path = f"{log_dir}/depth_{cam_name}.jpg"

        plt.imshow(rgb)
        plt.savefig(rgb_path)
        plt.clf()

        plt.imshow(np.clip(depth, 0, 1))
        plt.savefig(depth_path)
        plt.clf()

    return rgb, depth


def plot_3d_points_with_animation(point_set, save_file=None):
    """
    pointset: numpy.ndarray arranged as (t, num_points, 3)
    save_file: path to save the animation to
    """
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim3d(-0.1, 0.5)
    ax.set_xlim3d(-0.1, 0.1)
    ax.set_ylim3d(-0.1, 0.1)
    title = ax.set_title('3d Animation of sensor falling')
    graph = ax.scatter(point_set[0, :, 0], point_set[0, :, 1], point_set[0, :, 2])
    def update_graph(num):
        graph._offsets3d = (point_set[num, :, 0], point_set[num, :, 1], point_set[num, :, 2])
    ani = animation.FuncAnimation(
        fig, update_graph, 200, blit=False
    )
    ani.save(save_file, writer=writer)


def generate_points_on_circle(center, radius):
    angles = np.linspace(0, 2*np.pi, 10)
    x = np.cos(angles)
    y = np.sin(angles)

    # move them to the desired center
    x += center[0]
    y += center[1]
    x *= radius
    y *= radius
    return np.c_[x, y]


def circle_pts(radius, angles):
    xs = radius*np.cos(angles)
    ys = radius*np.sin(angles)
    return np.c_[xs, ys]


def generate_axis_vector(from_pts, to_pt):
    """
        from_pts: numpy.ndarray of type (num, 3)
        to_pt: constant point
        computes vector from_pts -> to_pt
    """
    look_at_vector = from_pts - to_pt
    # normalize the vector
    normalizer = np.linalg.norm(look_at_vector, axis=1)
    normalizer = normalizer.reshape(-1, 1)
    look_at_vector = look_at_vector / normalizer
    return -look_at_vector


def save_pts(all_t_geom_xpos):
    def custom_draw_geometry_with_key_callback(pcd):
        def capture_image(vis):
            image = vis.capture_screen_float_buffer()
            plt.imshow(np.asarray(image))
            plt.show()
            return False

        key_to_callback = {}
        key_to_callback[ord(".")] = capture_image
        o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

    for t in range(len(all_t_geom_xpos)):
        pts = all_t_geom_xpos[t]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        custom_draw_geometry_with_key_callback(pcd)


def rotate_along_x(angle):
    angle = np.deg2rad(angle)
    rot_mat = np.eye(3)
    rot_mat[1, 1] = np.cos(angle)
    rot_mat[1, 2] = -np.sin(angle)
    rot_mat[2, 1] = np.sin(angle)
    rot_mat[2, 2] = np.cos(angle)
    return rot_mat


def render_sensor(point_set,
        render_sensor_path="/Users/macbookpro15/Documents/mujoco_hand_exps/data/sensor_render"):
    """
    pointset: it is collectiono of sensor points for all timesteps
    """
    # first take one of the point, subtract the center from it which
    # I know is the 0-th position out of the 220 points
    # form the mesh from this
    if not os.path.exists(render_sensor_path):
        os.makedirs(render_sensor_path)
    time_steps = len(point_set)
    for t in range(time_steps):
        sensor = trimesh.load_mesh(f'../data/mesh_dir/mesh_{t}_out/mc_mesh_out.ply')
        sensor_mesh = Mesh.from_trimesh(sensor)
        # Light for the scene
        direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
        spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                           innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
        point_l = PointLight(color=np.ones(3), intensity=10.0)

        # add camera to the scene
        cam = PerspectiveCamera(yfov=(np.pi / 3.0))
        cam_pose = np.array([
            [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
            [1.0, 0.0,           0.0,           0.0],
            [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
            [0.0,  0.0,           0.0,          1.0]
        ])

        # create the scene
        scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
        point_mesh_node = scene.add(sensor_mesh)
        direc_l_node = scene.add(direc_l, pose=cam_pose)
        spot_l_node = scene.add(spot_l, pose=cam_pose)
        cam_node = scene.add(cam, pose=cam_pose)
        print('rendering the scene offline')
        r = OffscreenRenderer(viewport_width=640, viewport_height=480)
        color, depth = r.render(scene)
        r.delete()

        plt.figure()
        plt.imshow(color)
        plt.savefig(f'{render_sensor_path}/img_{t}.jpg')


def non_blocking_visualization(all_t_geom_xpos, save_img=False, log_dir=None,
                               render_option_file=None):
    """
    I see the perspective problem here, so if I remove the translation
    it should be fine. Removing translation by subtracting the 0-th element
    from each of the points. This should be used.
    """
    if save_img:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    pcd = o3d.geometry.PointCloud()

    rotmat = np.asarray([[1, 0, 0], [0, 0, -1], [0, -1, 0]])
    pts = all_t_geom_xpos[0]
    pts = np.dot(rotmat, pts.T).T
    pts = pts - pts[0]
    pcd.points = o3d.utility.Vector3dVector(pts)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.5)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(frame)
    vis.get_render_option().load_from_json(render_option_file)
    if save_img:
        vis.capture_screen_image(f"{log_dir}/img_{0}.jpg")

    for t in range(1, len(all_t_geom_xpos)):
        pts = all_t_geom_xpos[t]
        pts = np.dot(rotmat, pts.T).T
        pts = pts - pts[0]
        pcd.points = o3d.utility.Vector3dVector(pts)

        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        if save_img:
            vis.capture_screen_image(f"{log_dir}/img_{t}.jpg")

    vis.destroy_window()


def create_motion_from_pcds(p1, p2, t_step=None, save_dir=None):
    """

    :param p1: positions of sensor geoms at timestep 1
    :param p2: positions of sensor geoms at timestep 2
    :return: open3d.geometry.Lineset
    """
    # I am setting the seed here will it affect seed everywhere
    c = np.random.uniform(size=p1.shape[0])
    colors = np.c_[c, c, c]
    correpondences = [(i, i) for i in range(p1.shape[0])]
    pcd1 = o3d.geometry.PointCloud()
    p1 = p1 - p1[0]
    pcd1.points = o3d.utility.Vector3dVector(p1)

    pcd2 = o3d.geometry.PointCloud()
    p2 = p2 - p2[0]
    pcd2.points = o3d.utility.Vector3dVector(p2)

    lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd1, pcd2, correpondences)
    lineset.colors = o3d.utility.Vector3dVector(colors)

    return lineset


def make_lineset_for_all_timesteps(pointset, save_dir=None):
    """

    :param pointset: positions of sensor geoms for all timesteps
    :param save_dir: where do you want to save the resutling lineset images
    :return: None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timesteps = len(pointset)
    linesets = list()
    for t in range(timesteps):
        p1 = pointset[0]
        p2 = pointset[t]
        l = create_motion_from_pcds(p1, p2, t_step=t, save_dir=save_dir)
        linesets.append(l)
    o3d.visualization.draw_geometries(linesets)
    print("done")


def non_blocking_visualization_mesh(mesh_dir):
    """
    mesh_dir: is the collection of base mesh dir
    """
    # timesteps = 80 # TODO: Remove the hardcoding
    # meshes_path = [os.path.join(mesh_dir, f"mesh_{i}_out/mc_mesh_out.ply") for i in range(timesteps)]
    # for t in range(timesteps):
    #     mesh = o3d.geometry.
    # from IPython import embed; embed()
    raise NotImplementedError


def convert_to_videos(img_dir, name_of_video, rate):
    command = f"ffmpeg -r {rate} -f image2 -s 1280x960 -i {img_dir}/img_%d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p {name_of_video}.mp4"
    print(command)
    os.system(command)
    print("done")
    file_path = Path(os.path.join(img_dir, f"{name_of_video}.mp4"))
    if file_path.is_file():
        return True
    return False


def sym(w):
    return w.dot(inv(sqrtm(w.T.dot(w))))


def get_quaternion(z_axis, world_up):
    """
    z_axis = numpy.ndarray(n_pts, 3)
    world_up = axis representing the z axis
    """
    world_up = np.tile(world_up, len(z_axis)).reshape(len(z_axis), 3)
    side_axis = np.cross(world_up, z_axis)
    side_axis = side_axis / np.linalg.norm(side_axis, axis=1).reshape(-1, 1)
    up_axis = np.cross(z_axis, side_axis)

    # TODO: find a better way to do this
    rot_mat = np.zeros((len(z_axis), 4, 4))
    quats = list()
    for i in range(len(rot_mat)):
        rot_mat[i, :3, 0] = side_axis[i]
        rot_mat[i, :3, 1] = up_axis[i]
        rot_mat[i, :3, 2] = z_axis[i]
        rot_mat[i, 3, 3] = 1
        rot_mat[i] = sym(rot_mat[i])
        quats.append(transformations.quaternion_from_matrix(rot_mat[i]))
    return np.stack(quats)


def generate_new_cameras(radius, center, lookat_vector, height, jitter_z=False, num_pts=50):
    # generate points on the circle
    angle = np.linspace(0, 2*np.pi, num=num_pts) # in radians
    # angle = np.asarray([0, np.pi/2., np.pi, 3*np.pi/2.])
    xy_pts = circle_pts(radius, angle)
    plt.scatter(xy_pts[:, 0], xy_pts[:, 1], c='b')
    # plt.show()

    # xyz_points
    xyz_points = np.c_[xy_pts[:, 0], xy_pts[:, 1], height*np.ones(len(xy_pts))]
    xyz_points[:, 0] += center[0]
    xyz_points[:, 1] += center[1]
    if jitter_z:
        xyz_points[:, 2] += (0.02*np.random.normal(size=num_pts))
    # generate the z-axis for each of these
    z_vector = xyz_points - lookat_vector
    z_axis = z_vector / np.linalg.norm(z_vector, axis=1).reshape(-1, 1)
    # now from this I will also generate the other two axis and quaternion
    quat = get_quaternion(z_axis, world_up=np.asarray([0., 0., 1.]))
    return xyz_points, quat

def get_axis_directions_out(npy_file):
    splits = npy_file[:-4].split('/')
    last = splits[-1]
    b = last.split('_')
    axis_direction = [b[-2], b[-1]]
    return axis_direction


def custom_draw_geometry_load_option(pcd, render_option_file=None):
    print('render option file path: ', render_option_file)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json(render_option_file)
    vis.run()
    vis.destroy_window()


def make_and_visualize_pcd(pts, render_option_file=None):
    """
    make open3d visualizer and visualize points
    :param pts: numpy.ndarray(npts, 3)
    :param render_option_file: file which specifies the size of points and other options
    :return: None
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    custom_draw_geometry_load_option(pcd, render_option_file="render_options_open3d.json")


def overlay_images(move_img_path, sensor_vis_capsule, overlay_store_dir):
    """

    :param move_img_path: this contains images for mujoco movement
    :param sensor_vis_capsule: this contains sensor animation images
    :param overlay_store_dir: where to store the overlayed images
    :return: overlayed images in the specified directory
    """
    if not os.path.exists(move_img_path):
        raise FileNotFoundError('movement directory does not exists')

    if not os.path.exists(sensor_vis_capsule):
        raise FileNotFoundError('sensor visualization directory does not exists')

    # make the dir for the overlayed images
    if not os.path.exists(overlay_store_dir):
        os.makedirs(overlay_store_dir)

    # now go through each image and do the overlay
    all_move_images = [os.path.join(move_img_path, img) for img in os.listdir(move_img_path) if 'jpg' in img]
    all_sensor_images = [os.path.join(sensor_vis_capsule, img) for img in os.listdir(sensor_vis_capsule) if 'jpg' in img]

    all_move_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    all_sensor_images.sort(key=lambda f: int(''.join((filter(str.isdigit, f)))))

    # load a imagem and get parameters from it
    a_imgm = Image.open(pathlib.Path(all_move_images[0]))
    width_resize, height_resize = a_imgm.size[0]//2, a_imgm.size[1]//2
    width, height = a_imgm.size[0], a_imgm.size[1]

    # zip them and go through them
    for i, (mimg, simg) in enumerate(zip(all_move_images, all_sensor_images)):
        # load and visualize the two images
        imgm = Image.open(pathlib.Path(mimg))
        imgs = Image.open(pathlib.Path(simg))
        imgs_resized = imgs.resize((width_resize, height_resize), Image.BILINEAR)
        imgm.paste(imgs_resized, (width-width_resize, height-height_resize))

        # now save the image in the overlay_store_dir
        imgm.save(f'{overlay_store_dir}/img_{i}.jpg')


def overlay_deformations_and_positions(img_dir, geom_pos_dir, video_dir):
    """
    I make the temporary overlay dir to store the images and I also make temporary
    sensor animation directory to store the sensor positions
    :param img_dir: where are the images for the movement, remember to only include jpgs
    :param geom_pos_dir: what is the base directory for geom positions
    :param video_dir: where will all the videos be store
    :return: makes the overlayed video in the directory
    """
    # create a temp directory for overlay
    overlay_directory = "/Users/macbookpro15/Documents/mujoco_hand_exps/data/overlay_temp"
    if not os.path.exists(overlay_directory):
        os.makedirs(overlay_directory)

    # now make the temp directory for the sensor animations images
    sensor_anim_temp = "sensor_anim_temp"
    if not os.path.exists(sensor_anim_temp):
        os.makedirs(sensor_anim_temp)

    # finally make the video_dir #NOTE
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    all_npy_files = [os.path.join(geom_pos_dir, f) for f in os.listdir(geom_pos_dir) if ".npy" in f]
    for i in range(len(all_npy_files)):
        axis_direction = get_axis_directions_out(all_npy_files[i])
        npy_data = np.load(all_npy_files[i])
        non_blocking_visualization(npy_data, save_img=True, log_dir=sensor_anim_temp,
                                   render_option_file="render_options_open3d.json")
        move_img_path = os.path.join(img_dir, f"axis_{axis_direction[0]}_{axis_direction[1]}")
        overlay_images(move_img_path, sensor_anim_temp, overlay_directory)

        # okay now lets make the videos
        overlay_video_path = f"{video_dir}/overlayed_{axis_direction[0]}_{axis_direction[1]}"
        res = convert_to_videos(overlay_directory, overlay_video_path, rate=8)
        if not res:
            raise ValueError('this is false means the video was not made')


if __name__ == '__main__':
    # here will go the bowl stuff to test the above function
    bowl_positions_dir = "/Users/macbookpro15/Documents/mujoco_hand_exps/data/bowl_positions_circle"
    bowl_imgs_dir = "/Users/macbookpro15/Documents/bowl_upright"
    video_dir = "/Users/macbookpro15/Documents/mujoco_hand_exps/data/bowl_should_use_this"

    overlay_deformations_and_positions(bowl_imgs_dir, bowl_positions_dir, video_dir)
