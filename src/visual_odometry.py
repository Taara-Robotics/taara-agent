import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def pnp_ransac(
    points_3d: np.ndarray, points_2d: np.ndarray, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray 
):
    s, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not s or inliers is None:
        return np.eye(4), np.array([], dtype=np.int32)
    
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = tvec.ravel()
    T[:3, :3] = Rotation.from_rotvec(rvec.ravel()).as_matrix()
    T = np.linalg.inv(T)

    return T, inliers.flatten()


def rgbd_coordinates_to_3d_points(
    uv: np.ndarray, color: np.ndarray, depth: np.ndarray, depth_scale: float, camera_matrix: np.ndarray
):
    depth_uv = uv / np.array([color.shape[1], color.shape[0]]) * np.array([depth.shape[1], depth.shape[0]])
    z = bilinear_depth_interpolation(depth, depth_uv) / depth_scale
    x = z * (uv[:, 0] - camera_matrix[0, 2]) / camera_matrix[0, 0]
    y = z * (uv[:, 1] - camera_matrix[1, 2]) / camera_matrix[1, 1]

    return np.array((x, y, z), dtype=np.float32).T


def bilinear_depth_interpolation(depth: np.ndarray, uv: np.ndarray):
    u0 = uv[:, 0].astype(np.uint)
    v0 = uv[:, 1].astype(np.uint)

    up = uv[:, 0] - u0
    vp = uv[:, 1] - v0
    d0 = depth[v0, u0]
    d1 = depth[v0, u0 + 1]
    d2 = depth[v0 + 1, u0]
    d3 = depth[v0 + 1, u0 + 1]
    d = (1 - vp) * (d1 * up + d0 * (1 - up)) + vp * (d3 * up + d2 * (1 - up))

    return d


def kpts_3d_to_world(pose: np.ndarray, kpts_3d: np.ndarray):
    kpts_3d_homogeneous = np.concatenate(
        [kpts_3d, np.ones((kpts_3d.shape[0], 1))], axis=1
    )
    points_3d = pose @ kpts_3d_homogeneous.T
    return points_3d[:3].T


class VisualOdometry:
    def __init__(self, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray, depth_scale: float, depth_min: float, depth_max: float):
        self._camera_matrix = camera_matrix
        self._distortion_coefficients = distortion_coefficients
        self._depth_scale = depth_scale
        self._depth_min = depth_min
        self._depth_max = depth_max
        self._orb = cv2.ORB_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev = None
        self._pose = np.eye(4)
    
    def process_frame(self, color: np.ndarray, depth: np.ndarray):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        mask = np.logical_and(depth > self._depth_min*self._depth_scale, depth < self._depth_max*self._depth_scale).astype(np.uint8)
        mask = cv2.resize(mask, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
        kpts, descs = self._orb.detectAndCompute(gray, None)

        if descs is None:
            return self._pose

        if self._prev is None:
            self._prev = (kpts, descs)
            return self._pose
        
        prev_kpts, prev_descs = self._prev
        self._prev = (kpts, descs)

        matches = self._matcher.match(descs, prev_descs)

        if len(matches) < 4:
            return self._pose

        prev_points = rgbd_coordinates_to_3d_points(
            np.array([prev_kpts[m.trainIdx].pt for m in matches]),
            color,
            depth,
            self._depth_scale,
            self._camera_matrix,
        )

        T, inliers = pnp_ransac(
            prev_points,
            np.array([kpts[m.queryIdx].pt for m in matches]),
            self._camera_matrix,
            self._distortion_coefficients,
        )

        if len(inliers) < 4:
            return self._pose
        
        self._pose = self._pose @ T
        return self._pose


if __name__ == "__main__":
    # read camera matrix
    data_path = "../taara-slam/data/reconstruction"

    with open(f"{data_path}/camera_intrinsic.json", "r") as f:
        data = json.load(f)
        camera_matrix = np.array(data["intrinsic_matrix"]).reshape(3, 3).T
        distortion_coefficients = np.zeros(5)
        depth_scale = 1000.0
        depth_min = 0.25
        depth_max = 3.86

    color_filenames = sorted(glob.glob(f"{data_path}/color/*.jpg"))
    depth_filenames = sorted(glob.glob(f"{data_path}/depth/*.png"))

    # run VO
    vo = VisualOdometry(camera_matrix, distortion_coefficients, depth_scale, depth_min, depth_max)
    trajectory = []

    for color_filename, depth_filename in zip(color_filenames, depth_filenames):
        color = cv2.imread(color_filename)
        depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)

        pose = vo.process_frame(color, depth)
        trajectory.append(pose[:3, 3])
    
    # plot trajectory
    trajectory = np.array(trajectory)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
