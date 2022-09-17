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
from scipy.optimize import least_squares
import time

class Visual_Odometry():
    def __init__(self, data_dir, descriptor="ORB", num_features=3000, matcher="FLANN", show_matches=False):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(data_dir + '/calib.txt')
        self.images_l = self._load_images(data_dir + '/image_0')
        self.images_r = self._load_images(data_dir + '/image_1')
        self.show_matches = show_matches

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparities = [np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]

        if descriptor=="ORB":
            self.descriptor = cv2.ORB_create(num_features)
        elif descriptor=="SIFT":
            self.descriptor = cv2.SIFT_create(num_features)

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
            K_l (ndarray): Intrinsic Parameters of the left camera
            P_l (ndarray): Projection matrix of the left camera
            K_r (ndarray): Intrinsic Parameters of the right camera
            P_r (ndarray): Projection matrix of the right camera
        """
        with open(filepath, "r") as file:
            params = np.fromstring(file.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]

            params = np.fromstring(file.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]

        return K_l, P_l, K_r, P_r

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
    def _load_images(filepath):
        """
        Loads all the images in memory for faster computation
        # Should be changed for real-time code though
        Args:
            filepath (str): The file path to the images directory

        Returns:
            images (list): List of images in ndarray format
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]
        return images

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

    @staticmethod
    def get_idxs(q, disp, min_disp, max_disp):
        q_idx = q.astype(int)
        disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
        mask = np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        return disp, mask

    def get_matches(self, img1, img2):
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
        kp1, des1 = self.descriptor.detectAndCompute(img1, None)
        kp2, des2 = self.descriptor.detectAndCompute(img2, None)

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

        # TODO: Remove this for something better
        draw_params = dict(matchColor = -1, singlePointColor = None, matchesMask = None, flags = 2)

        if self.show_matches:
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, good ,None,**draw_params)
            cv2.imshow("image", img3)
            cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

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

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)
        Args:
            q1 (ndarray) (n,2): Feature points in i-1'th left image.
            q2 (ndarray) (n,2): Feature points in i'th left image.
            disp1 (ndarray) (H x W): Disparity i-1'th image per.
            disp2 (ndarray) (H x W): Disparity i'th image per.
            min_disp (float): The minimum disparity
            max_disp (float): The maximum disparity

        Returns:
            q1_l (ndarray) (n,2): Feature points in i-1'th left image.
            q1_r (ndarray) (n,2): Feature points in i-1'th right image.
            q2_l (ndarray) (n,2): Feature points in i'th left image.
            q2_r (ndarray) (n,2): Feature points in i'th right image.
        """

        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = self.get_idxs(q1, disp1, min_disp, max_disp)
        disp2, mask2 = self.get_idxs(q2, disp2, min_disp, max_disp)

        # Combine the masks
        in_bounds = np.logical_and(mask1, mask2)

        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]

        # Calculate the right feature points
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2

        return q1_l, q1_r, q2_l, q2_r

    def get_pose(self, i):
        """
        Calculate the transformation matrix
        Args:
            q1 (ndarray): The good key-point matches in i-1'th image
            q2 (ndarray): the good key-point matches in the i'th image

        Returns:
            transformation_matrix (ndarray): The transformation matrix containing rotation
            and translation data
        """

        # Step 1: Get the two consecutive images from loaded images
        img1_l, img2_l = self.images_l[i-1:i+1]

        # Step 2: Get the matches
        # if self.descriptor=="FAST":
        #     kp1_l = self.get_keypoints(img1_l, 10, 20)
        #
        #     tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        tp1_l, tp2_l = self.get_matches(img1_l, img2_l)

        # Step 3: Calculate the disparities
        self.disparities.append(np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

        # Step 4: Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])

        # Step 5: Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Step 6: Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)

        return transformation_matrix

def main(render=False, pass_gt=False):
    data_dir = "/home/NEU_Courses/EECE5554/Final_Project/Dataset/KITTI_sequences/00"
    # vo = Visual_Odometry(data_dir, descriptor="SIFT", matcher="BFM")
    vo = Visual_Odometry(data_dir)

    gt_poses = [np.eye(4)]
    if pass_gt:
        gt_poses = Visual_Odometry._load_poses(os.path.join(data_dir, 'poses.txt'))

    gt_path = []
    estimated_path = []
    cur_pose = 0

    for i in tqdm.tqdm(range(len(vo.images_l)), unit="pose"):
        if i == 0:
            cur_pose = gt_poses[0]
        else:
            transformation = vo.get_pose(i)
            cur_pose = np.matmul(cur_pose, transformation)

        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        if pass_gt:
            gt_path.append((gt_poses[i][0, 3], gt_poses[i][2, 3]))

        if render:
            utils.plot_poses_realtime(estimated_path, gt_path, pass_gt)

    np.save("estimated path", np.array(estimated_path))
    np.save("gt path", np.array(gt_path))
    if render:
        plt.show()

if __name__ == "__main__":
    main(render=False, pass_gt=True)