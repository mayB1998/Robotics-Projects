import sys
import glob
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import time

def load_image(path: str) -> np.ndarray:
    return cv2.imread(path)[:, :, ::-1]

def get_matches(pic_a: np.ndarray, pic_b: np.ndarray, n_feat: int) -> (np.ndarray, np.ndarray):
    """Get unreliable matching points between two images using SIFT.

    You do not need to modify anything in this function, although you can if
    you want to.

    Args:
        pic_a: a numpy array representing image 1.
        pic_b: a numpy array representing image 2.
        n_feat: an int representing number of matching points required.

    Returns:
        pts_a: a numpy array representing image 1 points.
        pts_b: a numpy array representing image 2 points.
    """

    # TODO: Remove these two lines.
    pic_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    pic_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp_a, desc_a = sift.detectAndCompute(pic_a, None)
    kp_b, desc_b = sift.detectAndCompute(pic_b, None)
    dm = cv2.BFMatcher(cv2.NORM_L2)
    matches = dm.knnMatch(desc_b, desc_a, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < n.distance * 0.8:
            good_matches.append(m)
    pts_a = []
    pts_b = []
    for m in good_matches[: int(n_feat)]:
        pts_a.append(kp_a[m.trainIdx].pt)
        pts_b.append(kp_b[m.queryIdx].pt)

    return np.asarray(pts_a), np.asarray(pts_b)

def get_vo(data_path, focal = 718.8560, pp = (607.1928, 185.2157)):
    image_paths = sorted(glob.glob(data_path+"/*.png"))
    num_images = len(image_paths)

    poses_wTi = []
    poses_wTi += [np.eye(4)]

    for i in range(num_images - 1):
        img_i1 = load_image(image_paths[i])
        img_i2 = load_image(image_paths[i + 1])

        # SIFT version, estimate F
        pts_a, pts_b = get_matches(img_i1, img_i2, n_feat=int(4e3))

        i2_E_i1, _ = cv2.findEssentialMat(pts_a, pts_b, focal, pp, cv2.RANSAC, 0.999, 1.0, None)

        # # ORB version, estimate E
        # i2_F_i1, i2_E_i1, inliers_a, inliers_b = get_matches_ORB(img_i1, img_i2, K, fmat=True)
        # i2_E_i1 = get_emat_from_fmat(i2_F_i1, K1=K, K2=K)

        # _num_inlier, i2Ri1, i2ti1, _ = cv2.recoverPose(i2_E_i1, inliers_a, inliers_b)
        # _, R, t, _ = cv2.recoverPose(i2_E_i1, pts_b, pts_a, R, t, focal, pp,None)
        _, i2Ri1, i2ti1, _ = cv2.recoverPose(i2_E_i1, pts_b, pts_a)

        # form SE(3) transformation
        i2Ti1 = np.eye(4)
        i2Ti1[:3, :3] = i2Ri1
        i2Ti1[:3, 3] = i2ti1.squeeze()

        # use previous world frame pose, to place this camera in world frame
        # assume 1 meter translation for unknown scale (gauge ambiguity)
        wTi1 = poses_wTi[-1]
        i1Ti2 = np.linalg.inv(i2Ti1)
        wTi2 = wTi1 @ i1Ti2
        poses_wTi += [wTi2]

        r = Rotation.from_matrix(i2Ri1.T)
        rz, ry, rx = r.as_euler("zyx", degrees=True)
        print(f"Rotation about y-axis from frame {i} -> {i + 1}: {ry:.2f} degrees")

    return poses_wTi


def plot_poses(poses_wTi, figsize=(7, 8)):
    """
    Poses are wTi (in world frame, which is defined as 0th camera frame)
    """
    axis_length = 0.5

    num_poses = len(poses_wTi)

    _, ax = plt.subplots(figsize=figsize)

    for i, wTi in enumerate(poses_wTi):
        wti = wTi[:3, 3]

        # assume ground plane is xz plane in camera coordinate frame
        # 3d points in +x and +z axis directions, in homogeneous coordinates
        posx = wTi @ np.array([axis_length, 0, 0, 1]).reshape(4, 1)
        posz = wTi @ np.array([0, 0, axis_length, 1]).reshape(4, 1)

        # ax.plot([wti[0], posx[0]], [wti[2], posx[2]], "b", zorder=1)
        # ax.plot([wti[0], posz[0]], [wti[2], posz[2]], "k", zorder=1)

        ax.plot(wti[0], wti[2], 40, marker=".", color='g', zorder=2)

    plt.axis("equal")
    plt.title("Egovehicle trajectory")
    plt.xlabel("x camera coordinate (of camera frame 0)")
    plt.ylabel("z camera coordinate (of camera frame 0)")
    plt.show()

if __name__=="__main__":
    data_path = "/home/NEU_Courses/EECE5554/Final_Project/Dataset/KITTI_sequences/test/image_0/"
    poses = get_vo(data_path)

    plot_poses(poses)