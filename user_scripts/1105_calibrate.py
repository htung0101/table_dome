import os
import sys
import numpy as np
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers

def findTransform(P,Q):
    '''
    Finds best rigid body transform between 2 sets of points
    Assuming the two points are in correspondence
    Then the optimal transformation is given by the following
    1. Subtract the centroid from the pointsets
    2. Get the covariance matrix.
    3. Then do the SVD on the pointsets.
    4. Then optimal rotation is UV^T and translation is computed using rotation
       and the centroids
    '''
    # YOUR CODE GOES  HERE
    # compute the centroids
    P_mean = np.mean(P, axis=0)
    Q_mean = np.mean(Q, axis=0)

    assert P_mean.shape == (3,), "Mean computation is not right"
    assert Q_mean.shape == (3,), "Mean computation for Q is not right"

    centered_P = P - P_mean
    centered_Q = Q - Q_mean

    # now compute the covariance matrix.
    cov = np.dot(centered_P.T, centered_Q)

    assert cov.shape == (3, 3), "convariance computation is not right"

    # do the svd of the covariance matrix.
    u, s, vh = np.linalg.svd(cov, full_matrices=True)

    R = np.dot(vh.T, u.T)
    if np.linalg.det(R) < 0:
        print('reflection detected')
        vh[2, :] = vh[2, :]
        R = np.dot(vh.T, u.T)
    t = Q_mean - np.dot(R, P_mean)

    print('R=', R)
    print('T=', t)

    # QUANTITAVE CHECK, check the residual error for each point
    transformed_p = np.dot(R, P.T).T + t
    print(transformed_p)
    # import IPython;IPython.embed()

    error = np.mean(np.linalg.norm(transformed_p - Q, axis=1))
    print('error is {0}'.format(error))

    return R, t


def callback1(data):
    x = data.markers[0].pose.pose.position.x
    y = data.markers[0].pose.pose.position.y
    z = data.markers[0].pose.pose.position.z

    # get the orientation
    qx = data.markers[0].pose.pose.orientation.x
    qy = data.markers[0].pose.pose.orientation.y
    qz = data.markers[0].pose.pose.orientation.z
    qw = data.markers[0].pose.pose.orientation.w

    global t_1
    t_1 = np.asarray([x, y, z])


def callback2(data):
    x = data.markers[0].pose.pose.position.x
    y = data.markers[0].pose.pose.position.y
    z = data.markers[0].pose.pose.position.z

    # get the orientation
    qx = data.markers[0].pose.pose.orientation.x
    qy = data.markers[0].pose.pose.orientation.y
    qz = data.markers[0].pose.pose.orientation.z
    qw = data.markers[0].pose.pose.orientation.w

    global t_2
    t_2 = np.asarray([x, y, z])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_data', action='store_true', default=False)
    parser.add_argument('--n_points', type=int, default=10)

    args = parser.parse_args()

    if args.collect_data:
        rospy.init_node('whatever', anonymous=True)

        rospy.Subscriber('ar_1/ar_pose_marker', AlvarMarkers, callback1)
        rospy.Subscriber('ar_2/ar_pose_marker', AlvarMarkers, callback2)
        rospy.sleep(2)

        points = []
        # import IPython;IPython.embed()
        from time import sleep
        for i in range(args.n_points):
            raw_input('Press Enter to record points {0}/{1}...'.format(i+1, args.n_points))
            points.append([t_1, t_2])
            print(t_1, t_2)
            sleep(0.3)

        points = np.stack(points)
        np.save('points.npy', points)

    else:
        points = np.load('points.npy')

    points_A = points[:, 0, :] # cam3 
    points_B = points[:, 1, :] # cam4
    R, t = findTransform(points_A, points_B)
    T_mat = np.eye(4)
    T_mat[:3, :3] = R
    T_mat[:3, 3] = t
    np.save('cam4_T_cam3.npy', T_mat)

if __name__ == '__main__':
    main()