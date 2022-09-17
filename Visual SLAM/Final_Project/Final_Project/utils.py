"""
Description:
    Utils file containing functions for plotting the data and other stuff
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_poses(poses_wTi, poses_wTi_gt, figsize=(7, 8)):
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

        ax.plot([wti[0], posx[0]], [wti[2], posx[2]], "b", zorder=1)
        ax.plot([wti[0], posz[0]], [wti[2], posz[2]], "k", zorder=1)

        ax.scatter(wti[0], wti[2], 40, marker=".", color='g', zorder=2)

        # ground-truth information
        if len(poses_wTi_gt) > 0:
            wTi_gt = poses_wTi_gt[i]
            wti_gt = wTi_gt[:3, 3]
            posx = wTi_gt @ np.array([axis_length, 0, 0, 1]).reshape(4, 1)
            posz = wTi_gt @ np.array([0, 0, axis_length, 1]).reshape(4, 1)

            ax.plot([wti_gt[0], posx[0]], [wti_gt[2], posx[2]], "m", zorder=1)
            ax.plot([wti_gt[0], posz[0]], [wti_gt[2], posz[2]], "c", zorder=1)

            ax.scatter(wti_gt[0], wti_gt[2], 40, marker=".", color='r', zorder=2)

    plt.axis("equal")
    plt.title("Egovehicle trajectory")
    plt.xlabel("x camera coordinate (of camera frame 0)")
    plt.ylabel("z camera coordinate (of camera frame 0)")

def plot_poses_realtime(poses, poses_gt, plot_gt=True):
    poses_gt = np.array(poses_gt)
    poses = np.array(poses)

    plt.cla()
    if plot_gt:
        plt.plot(poses_gt[:, 0], poses_gt[:, 1], label="Ground Truth")

    plt.plot(poses[:, 0], poses[:, 1], label="Estimated trajectory")
    # plt.xlim([-100,100])
    # plt.ylim([0,200])
    plt.legend()
    plt.title("Estimated Trajectory using VO")
    plt.xlabel("metres")
    plt.ylabel("metres")
    plt.grid()
    # plt.axis('equal')
    plt.pause(0.001)

def read_tum_data(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        data = []
        for line in lines:
            point = line.split(" ")
            point = [float(point[1]), float(point[2])]
            data.append(point)

    return data

def plot_tum(data):
    data = np.array(data)
    plt.scatter(data[:,0], data[:,1])
    plt.show()

if __name__=="__main__":
    ep = np.load("/home/NEU_Courses/EECE5554/Final_Project/estimated_path_it3_mono_nuance_SIFT_3000.npy")
    gp = np.load("gt_path.npy")
    plot_poses_realtime(ep,gp, plot_gt=False)
    plt.show()
    # file_path = "/home/catkin_ws_orb/src/ORB_SLAM3/CameraTrajectory.txt"
    # data = read_tum_data(file_path)
    # print(data)
    # plot_tum(data)
