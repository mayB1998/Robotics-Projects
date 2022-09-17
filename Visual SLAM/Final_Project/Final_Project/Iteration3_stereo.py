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
        self.show_matches = show_matches
        self.descriptor_name = descriptor
        self.num_features = num_features

        self.keypoints1 = None
        self.descriptors1 = None

        block = 11
        P1 = 8*3*36
        P2 = 32*3*36
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=block, P1=P1, P2=P2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.disparity_old = None

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
        kp1 = self.keypoints1
        des1 = self.descriptors1
        # kp1, des1 = self.descriptor.detectAndCompute(img1, None)
        kp2, des2 = self.descriptor.detectAndCompute(img2, None)

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

        draw_params = dict(matchColor = -1, singlePointColor = None, matchesMask = None, flags = 2)

        if self.show_matches:
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, good ,None,**draw_params)
            cv2.imshow("image", img3)
            cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def calc_depth_map(self, disp_left, b=0.54, rectified=True):
        """
        Calculates the depth map from the disparity
        Args:
            disp_left: disparity map of left camera
            k_left: intrinsic matrix for left camera
            b: baseline of the stereo camera
            rectified: variable to define if the images are rectified or not

        Returns:
            depth_map (HxW): depth map calculated from disparity
        """
        # Get focal length of x-axis for left camera
        f = self.K_l[0][0]
        # print(f)

        # Avoid instability and division by zero
        disp_left[disp_left == 0.0] = 0.1
        disp_left[disp_left == -1.0] = 0.1

        # Make empty depth map then fill with depth
        # print(f"disp_left.shape: {disp_left.shape}")
        # print(f"b.shape: {b}")
        depth_map = f * b / disp_left

        # print(disp_left)
        # print(depth_map)
        # # cv2.imshow("image", disp_left)
        # # cv2.waitKey(1000)
        # plt.imshow(depth_map)
        # plt.show()

        # print(f"max_depth: {np.max(depth_map)}")
        # print(f"min_depth: {np.min(depth_map)}")
        return depth_map

    def calc_3D(self, depth_map, q1, q2, max_depth=3000):
        """
        Calculates the 3D position of the key-points from depth map and 2D key-points
        Args:
            depth_map (HxW): Depth map of the left image
            q1 (n,2): Key-points in the i-1th image
            q2 (n,2): Key-points in the ith image
            max_depth (int): Max depth threshold to prevent erroneous measurements

        Returns:
            object_points (m,2): 3D locations of the detected key-points
            q2 (m,2): 2D locations of the key-points with correct disparity values
        """
        cx = self.K_l[0,2]
        cy = self.K_l[1,2]
        fx = self.K_l[0,0]
        fy = self.K_l[1,1]
        delete = []
        object_points = np.zeros((0,3))

        for i, (u,v) in enumerate(q1):
            z = depth_map[int(v), int(u)]
            if z>max_depth:
                delete.append(i)
                continue

            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])

        q2 = np.delete(q2, delete, 0)
        return object_points, q2

    def get_pose(self, img1_l, img2_l, img2_r):
        """
        Calculate the transformation matrix
        Args:
            q1 (ndarray): The good key-point matches in i-1'th image
            q2 (ndarray): the good key-point matches in the i'th image

        Returns:
            transformation_matrix (ndarray): The transformation matrix containing rotation
            and translation data
        """

        # Step 1: Get the matches
        tp1_l, tp2_l = self.get_matches(img1_l, img2_l)

        # Step 2: Calculate the disparities
        disparity = np.divide(self.disparity.compute(img2_l, img2_r).astype(np.float32), 16)
        depth_map = self.calc_depth_map(disparity)

        # Step 3: Calculate the 3D points
        q_3d, q_2d = self.calc_3D(depth_map, tp1_l, tp2_l, max_depth=3000)

        # Step 4: Estimate the transformation matrix
        _, r, t, _ = cv2.solvePnPRansac(q_3d, q_2d, self.K_l, None)
        R, _ = cv2.Rodrigues(r)
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

def main(render=False, pass_gt=False):
    # data_dir = "/home/NEU_Courses/EECE5554/Final_Project/Dataset/KITTI_sequences/00"
    data_dir = "/home/Downloads/data"
    vo = Visual_Odometry(data_dir, descriptor="SIFT", matcher="BFM")
    # vo = Visual_Odometry(data_dir, matcher="BFM")

    gt_poses = [np.eye(4)]
    if pass_gt:
        gt_poses = Visual_Odometry._load_poses(os.path.join(data_dir, 'poses.txt'))

    gt_path = []
    estimated_path = []
    cur_pose = 0

    image_paths_l = [os.path.join(data_dir + "/image_0", file) for file in sorted(os.listdir(data_dir + "/image_0"))]
    image_paths_r = [os.path.join(data_dir + "/image_1", file) for file in sorted(os.listdir(data_dir + "/image_1"))]


    for i in tqdm.tqdm(range(10000), unit="pose"):
        if i == 0:
            cur_pose = gt_poses[0]
            img0_l = cv2.imread(image_paths_l[i], cv2.IMREAD_GRAYSCALE)
            vo.keypoints1, vo.descriptors1 = vo.descriptor.detectAndCompute(img0_l, None)
        else:
            img1_l = cv2.imread(image_paths_l[i], cv2.IMREAD_GRAYSCALE)
            img1_r = cv2.imread(image_paths_r[i], cv2.IMREAD_GRAYSCALE)
            # noinspection PyUnboundLocalVariable
            transformation = vo.get_pose(img0_l, img1_l, img1_r)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transformation))

            img0_l = img1_l

        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        if pass_gt:
            gt_path.append((gt_poses[i][0, 3], gt_poses[i][2, 3]))

        if render:
            utils.plot_poses_realtime(estimated_path, gt_path, pass_gt)

    np.save(f"estimated_path_it3_mono_nuance_{vo.descriptor_name}_{vo.num_features}", np.array(estimated_path))
    if pass_gt:
        np.save(f"gt_path", np.array(gt_path))
    if render:
        plt.show()

if __name__ == "__main__":
    main(render=False, pass_gt=False)