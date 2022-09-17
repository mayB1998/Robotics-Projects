"""
Description:
    This file basically runs Monocular Visual Odometry. The data inputs required are
    the data directory with images and camera calibration file.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import utils
import time


class Visual_Odometry():
    def __init__(self, data_dir, descriptor="ORB", num_features=3000, matcher="FLANN", show_matches=False):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.show_matches = show_matches
        self.show_keypoints = False
        self.num_features = num_features
        self.descriptor_name = descriptor
        self.keypoints1 = None
        self.descriptors1 = None
        self.data_dir = data_dir
        self.curr_P = self.P
        self.prev_R = np.eye(3)
        self.prev_t = np.zeros((3,1))

        # Assign a feature descriptor for the model
        if descriptor == "ORB":
            self.descriptor = cv2.ORB_create(num_features)
        elif descriptor == "SIFT":
            self.descriptor = cv2.SIFT_create(num_features)

        # Assign a feature matcher for the model
        if matcher == "FLANN":
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        elif matcher == "BFM":
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    @staticmethod
    def _load_calib(filepath):
        """
        Load the intrinsic matrix and calibration matrix of the camera
        Args:
            filepath (str): File path to the calibration file

        Returns:
            K (ndarray): Intrinsic Parameters of the camera
            P (ndarray): Projection matrix of the camera
        """
        with open(filepath, "r") as file:
            params = np.fromstring(file.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]

        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Load the Ground truth poses for the sequence
        Args:
            filepath (str): The file path of the poses file

        Returns:
            poses (ndarray): The Ground truth poses
        """
        poses = []
        with open(filepath, "r") as file:
            for line in file.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector
        Args:
            R (ndarray): The rotation matrix
            t (ndarray): The translation vector

        Returns:
            T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, img0, img1):
        """
        This function detect and compute key-points and descriptors from the i-1'th
        and i'th image using the descriptor defined in the class constructor
        Args:
            i (int): Current frame id

        Returns:
            q1 (ndarray): The good keypoint matches in i-1'th image
            q2 (ndarray): the good keypoint matches in the i'th image

        """
        # Find the key-points and descriptors with the descriptor
        # kp1, des1 = self.descriptor.detectAndCompute(img0, None)
        kp1 = self.keypoints1
        des1 = self.descriptors1

        kp2, des2 = self.descriptor.detectAndCompute(img1, None)

        self.keypoints1 = kp2
        self.descriptors1 = des2
        # Find matches
        matches = self.matcher.knnMatch(des1, des2, k=2)
        # Find good matches using the Lowe's ratio test
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor=-1, singlePointColor=None, matchesMask=None, flags=2)

        if self.show_matches:
            img3 = cv2.drawMatches(img1, kp1, img0, kp2, good, None, **draw_params)
            cv2.imshow("image", img3)
            cv2.waitKey(10)

        # Get the image points from the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        if self.show_keypoints:
            img3 = cv2.drawKeypoints(img1, kp2, None, color=(0, 0, 255))
            cv2.imshow("image", img3)
            cv2.waitKey(10)

        return q1, q2

    def get_pose(self, img0, img1):
        """
        Calculate the transformation matrix
        Args:
            q1 (ndarray): The good key-point matches in i-1'th image
            q2 (ndarray): the good key-point matches in the i'th image

        Returns:
            transformation_matrix (ndarray): The transfromation matrix containing rotation
            and translation data
        """
        # Get 2D matches
        q1, q2 = self.get_matches(img0, img1)

        # Essential matrix
        E, mask = cv2.findEssentialMat(q1, q2, self.K, cv2.RANSAC, 0.999, 1.0)

        # Decompose the Essential matrix into R and t
        # noinspection PyArgumentList
        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K, mask)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def run_vo(self, pass_gt=False , render=False):
        gt_poses = [np.eye(4)]
        if pass_gt:
            gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))

        gt_path = []
        estimated_path = []
        cur_pose = 0

        image_paths = [os.path.join(data_dir + "/image_0", file) for file in sorted(os.listdir(data_dir + "/image_0"))]

        for i in tqdm.tqdm(range(0, len(image_paths)), unit="frames"):
            if i == 0:
                cur_pose = gt_poses[0]
                img0 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
                self.keypoints1, self.descriptors1 = self.descriptor.detectAndCompute(img0, None)
            else:
                img1 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
                # noinspection PyUnboundLocalVariable
                transformation = self.get_pose(img0, img1)
                cur_pose = cur_pose @ np.linalg.inv(transformation)

                img0 = img1

            estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
            if pass_gt:
                # noinspection PyTypeChecker
                gt_path.append([gt_poses[i][0, 3], gt_poses[i][2, 3]])

            if render:
                utils.plot_poses_realtime(estimated_path, gt_path, pass_gt)

        np.save(f"estimated_path_it5_mono_nuance_{vo.descriptor_name}_{vo.num_features}", np.array(estimated_path))
        np.save(f"gt_path", np.array(gt_path))

        if render:
            plt.show()

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals
        Args:
            dof (ndarray) (6,1): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation.
            q1 (ndarray) (n, 2): Feature points in i-1'th image.
            q2 (ndarray) (n, 2): Feature points in i'th image.
            Q1 (ndarray) (n, 3): 3D points from the i-1'th image.
            Q2 (ndarray) (n, 3): 3D points from the i'th image.

        Returns:
            residuals (ndarray) (2 * n * 2): The residuals.
        """

        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transformation = self._form_transf(R, t)
        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transformation)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transformation))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])
        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images

        Args:
            q1_l (ndarray) (n, 2): Feature points in i-1'th left image.
            q1_r (ndarray) (n, 2): Feature points in i-1'th right image.
            q2_l (ndarray) (n, 2): Feature points in i'th left image.
            q2_r (ndarray) (n, 2): Feature points in i'th right image.

        Returns:
            Q1 (ndarray)  (n, 3): 3D points seen from the i-1'th image.
            Q2 (ndarray)  (n, 3): 3D points seen from the i'th image.
        """

        # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])

        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix by using the 2D and 3d points
        Args:
            q1 (ndarray) (n,2): Feature points in i-1'th image.
            q2 (ndarray) (n,2): Feature points in i'th image.
            Q1 (ndarray) (n,3): 3D points seen from the i-1'th image.
            Q2 (ndarray) (n,3): 3D points seen from the i'th image.
            max_iter (int): The maximum number of iterations for Least squares optimization

        Returns:
            transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0
        out_pose = []
        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1

            # If we don't find any better result in early_termination_threshold iterations
            if early_termination == early_termination_threshold:
                print("Early termination")
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)

        return transformation_matrix

    def get_pose_with_bundle_adjustment(self, img0, img1):
        """
        Calculate the transformation matrix
        Args:
            q1 (ndarray): The good key-point matches in i-1'th image
            q2 (ndarray): the good key-point matches in the i'th image

        Returns:
            transformation_matrix (ndarray): The transfromation matrix containing rotation
            and translation data
        """
        # Get 2D matches
        q1, q2 = self.get_matches(img0, img1)

        # Essential matrix
        E, mask = cv2.findEssentialMat(q1, q2, self.K, cv2.RANSAC, 0.999, 1.0)

        # Decompose the Essential matrix into R and t
        # noinspection PyArgumentList
        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K, mask)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))

        # Get 3D points
        Q = cv2.triangulatePoints(self.curr_P, self.K*transformation_matrix[:3], q1.T, q2.T)
        # Un-homogenize the 3D points
        Q = np.transpose(Q[:3]/Q[3])

        # Update the current Projection matrix
        self.prev_R = self.prev_R * R
        self.prev_t = self.prev_t + t
        self.curr_P = self.K*self._form_transf(self.prev_R, self.prev_t)

        return transformation_matrix


if __name__ == "__main__":
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Mono_VO version: 5")

    data_dir = "/home/NEU_Courses/EECE5554/Final_Project/Dataset/KITTI_sequences/00"
    # data_dir = "/home/Downloads/data"
    # vo = Visual_Odometry(data_dir, descriptor="SIFT", matcher="BFM")
    vo = Visual_Odometry(data_dir)  # , matcher="BFM")
    vo.run_vo()

