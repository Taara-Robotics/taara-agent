import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation


class RgbdOdometry:
    def __init__(self, width: int, height: int, camera_matrix: np.ndarray, depth_scale: float, depth_min: float, depth_max: float):
        self._depth_scale = depth_scale
        self._options = o3d.pipelines.odometry.OdometryOption(
            # iteration_number_per_pyramid_level=o3d.open3d.utility.IntVector([4, 2, 1]),
            depth_min=depth_min,
            depth_max=depth_max,
        )
        self._intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, camera_matrix
        )

        self._pose = np.eye(4)
        self._prev = None

        print(self._options)
        
        
    def process_frame(self, color: np.ndarray, depth: np.ndarray):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=self._depth_scale,
            depth_trunc=self._options.depth_max,
            convert_rgb_to_intensity=False
        )

        if self._prev is None:
            self._prev = rgbd_image
            return self._pose

        s, T, I =  o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd_image, self._prev, self._intrinsic, np.eye(4), o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), self._options
        )
        
        if not s:
            return self._pose
        
        self._pose = self._pose @ T
        self._prev = rgbd_image

        return self._pose

if __name__ == "__main__":
    # read camera matrix
    data_path = "../taara-slam/data/reconstruction"

    with open(f"{data_path}/camera_intrinsic.json", "r") as f:
        data = json.load(f)
        width = 256#data["width"]
        height = 128#data["height"]
        camera_matrix = np.array(data["intrinsic_matrix"]).reshape(3, 3).T / data["width"] * width
        depth_scale = 1000.0
        depth_min = 0.25
        depth_max = 3.86

    color_filenames = sorted(glob.glob(f"{data_path}/color/*.jpg"))
    depth_filenames = sorted(glob.glob(f"{data_path}/depth/*.png"))

    # run odometry
    vo = RgbdOdometry(width, height, camera_matrix, depth_scale, depth_min, depth_max)
    trajectory = []

    for color_filename, depth_filename in zip(color_filenames, depth_filenames):
        color = cv2.imread(color_filename)
        depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)

        color = cv2.resize(color, (width, height))
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

        pose = vo.process_frame(color, depth)
        # break
        trajectory.append(pose[:3, 3])
        print(pose[:3,3])
    
    # plot trajectory
    trajectory = np.array(trajectory)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
