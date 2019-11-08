import numpy as np
import tf.transformations as tft

def main():
    # all the link transforms, 0indexing
    cam0coloroptical_frame_T_cam0link = np.eye(4)
    cam0coloroptical_frame_T_cam0link[:3, 3] = np.asarray([0.015, 0.001, 0.000])
    cam0coloroptical_frame_T_cam0link[:3, :3] = tft.quaternion_matrix([0.506, -0.494, 0.507, 0.492])[:3, :3]
    # cam0coloroptical_frame_T_cam0link[:3, :3] = pyquaternion.Quaternion([0.492, 0.506, -0.494, 0.507]).rotation_matrix

    cam1coloroptical_frame_T_cam1link = np.eye(4)
    cam1coloroptical_frame_T_cam1link[:3, 3] = np.asarray([0.015, 0.001, -0.000])
    cam1coloroptical_frame_T_cam1link[:3, :3] = tft.quaternion_matrix([0.507, -0.495, 0.501, 0.497])[:3, :3]
    # cam1coloroptical_frame_T_cam1link[:3, :3] = pyquaternion.Quaternion([0.497, 0.507, -0.495, 0.501]).rotation_matrix

    cam2coloroptical_frame_T_cam2link = np.eye(4)
    cam2coloroptical_frame_T_cam2link[:3, 3] = np.asarray([0.015, 0.001, 0.000])
    cam2coloroptical_frame_T_cam2link[:3, :3] = tft.quaternion_matrix([0.505, -0.495, 0.504, 0.495])[:3, :3]
    # cam2coloroptical_frame_T_cam2link[:3, :3] = pyquaternion.Quaternion([0.495, 0.505, -0.495, 0.504]).rotation_matrix

    cam3coloroptical_frame_T_cam3link = np.eye(4)
    cam3coloroptical_frame_T_cam3link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam3coloroptical_frame_T_cam3link[:3, :3] = tft.quaternion_matrix([0.506, -0.495, 0.504, 0.495])[:3, :3]
    # cam3coloroptical_frame_T_cam3link[:3, :3] = pyquaternion.Quaternion([0.495, 0.506, -0.495, 0.504]).rotation_matrix

    cam4coloroptical_frame_T_cam4link = np.eye(4)
    cam4coloroptical_frame_T_cam4link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam4coloroptical_frame_T_cam4link[:3, :3] = tft.quaternion_matrix([0.505, -0.496, 0.500, 0.499])[:3, :3]
    # cam4coloroptical_frame_T_cam4link[:3, :3] = pyquaternion.Quaternion([0.499, 0.505, -0.496, 0.500]).rotation_matrix

    cam5coloroptical_frame_T_cam5link = np.eye(4)
    cam5coloroptical_frame_T_cam5link[:3, 3] = np.asarray([0.015, 0.000, 0.000])
    cam5coloroptical_frame_T_cam5link[:3, :3] = tft.quaternion_matrix([0.505, -0.497, 0.503, 0.495])[:3, :3]
    # cam5coloroptical_frame_T_cam5link[:3, :3] = pyquaternion.Quaternion([0.495, 0.505, -0.497, 0.503]).rotation_matrix


    # 1-indexing
    ar_T_camlink1 = np.eye(4)
    ar_T_camlink1[:3, 3] = np.asarray([0.273531457485, -0.0116601092696, 0.515056866181])
    ar_T_camlink1[:3, :3] = tft.euler_matrix(-3.0944508747, 1.02325404017, 0.107969753333, 'rzyx')[:3, :3]

    ar_T_camlink2 = np.eye(4)
    ar_T_camlink2[:3, 3] = np.asarray([0.0293221158559, -0.215470391487, 0.377162423479])
    ar_T_camlink2[:3, :3] = tft.euler_matrix(1.66996067542, 0.898766747012, -0.0776489694977, 'rzyx')[:3, :3]

    ar_T_camlink3 = np.eye(4)
    ar_T_camlink3[:3, 3] = np.asarray([-0.313452228003, -0.240104942466, 0.561276816418])
    ar_T_camlink3[:3, :3] = tft.euler_matrix(0.79719381281, 1.03938842198, -0.236486700096, 'rzyx')[:3, :3]

    ar_T_camlink4 = np.eye(4)
    ar_T_camlink4[:3, 3] = np.asarray([-0.431590875458, 0.022398181249, 0.571518272949])
    ar_T_camlink4[:3, :3] = tft.euler_matrix(-0.364951012234, 0.992117141317, -0.2231357701, 'rzyx')[:3, :3]

    ar_T_camlink5 = np.eye(4)
    ar_T_camlink5[:3, 3] = np.asarray([-0.263625245653, 0.23208572851, 0.586816080247])
    ar_T_camlink5[:3, :3] = tft.euler_matrix(-0.991972902934, 1.08019557561, 0.20977623476, 'rzyx')[:3, :3]

    ar_T_camlink6 = np.eye(4)
    ar_T_camlink6[:3, 3] = np.asarray([0.126093519357, 0.242591670912, 0.390678960966])
    ar_T_camlink6[:3, :3] = tft.euler_matrix(-1.60058380551, 1.05379176473, 0.155491687996, 'rzyx')[:3, :3]

    extrinsics = list()
    for i in range(0, 6):
        extrinsics.append(eval('ar_T_camlink{0}'.format(i+1)).dot(np.linalg.inv(eval('cam{0}coloroptical_frame_T_cam{0}link'.format(i)))))

    extrinsics = np.stack(extrinsics)
    np.save('smartest_calibration.npy', extrinsics)

if __name__ == '__main__':
    main()