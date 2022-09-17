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
    def __init__(self, data_dir, descriptor="ORB", num_features=8000, matcher="FLANN", show_matches=False):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'image_0'))
        self.show_matches = show_matches
        self.num_features = num_features
        self.descriptor_name = descriptor

        # Assign a feature descriptor for the model
        if descriptor=="ORB":
            self.descriptor = cv2.ORB_create(num_features)
        elif descriptor=="SIFT":
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

    def get_matches(self, i):
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
        kp1, des1 = self.descriptor.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.descriptor.detectAndCompute(self.images[i], None)

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
        draw_params = dict(matchColor = -1, # draw matches in green color
                         singlePointColor = None,
                         matchesMask = None, # draw only inliers
                         flags = 2)

        if self.show_matches:
            img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1], kp2, good ,None,**draw_params)
            cv2.imshow("image", img3)
            cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculate the transformation matrix
        Args:
            q1 (ndarray): The good key-point matches in i-1'th image
            q2 (ndarray): the good key-point matches in the i'th image

        Returns:
            transformation_matrix (ndarray): The transfromation matrix containing rotation
            and translation data
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, cv2.RANSAC, 0.999, 1.0, None)

        # Decompose the Essential matrix into R and t
        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

def main(render=False, pass_gt=False):
    data_dir = "/home/NEU_Courses/EECE5554/Final_Project/Dataset/KITTI_sequences/00"
    # data_dir = "/home/Downloads/data"
    vo = Visual_Odometry(data_dir, descriptor="SIFT", matcher="BFM")
    # vo = Visual_Odometry(data_dir)

    gt_poses = [np.eye(4)]
    if pass_gt:
        gt_poses = Visual_Odometry._load_poses(os.path.join(data_dir, 'poses.txt'))

    gt_path = []
    estimated_path = []
    cur_pose = 0

    for i in tqdm.tqdm(range(len(vo.images)), unit="pose"):
        if i == 0:
            cur_pose = gt_poses[0]
        else:
            q1, q2 = vo.get_matches(i)
            transformation = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transformation))

        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        if pass_gt:
            gt_path.append([gt_poses[i][0, 3], gt_poses[i][2, 3]])

        if render:
            utils.plot_poses_realtime(estimated_path, gt_path, pass_gt)

    np.save(f"estimated_path_it2_mono_nuance_{vo.descriptor_name}_{vo.num_features}", np.array(estimated_path))
    np.save(f"gt_path", np.array(gt_path))
    if render:
        plt.show()

if __name__ == "__main__":
    main(render=True, pass_gt=True)