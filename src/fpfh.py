import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation

def draw_registration_result(source, target, transformation):
    source_temp = o3d.geometry.PointCloud(source)
    target_temp = o3d.geometry.PointCloud(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

# read camera matrix
data_path = "../taara-slam/data/reconstruction"

with open(f"{data_path}/camera_intrinsic.json", "r") as f:
    data = json.load(f)
    width = data["width"]
    height = data["height"]
    camera_matrix = np.array(data["intrinsic_matrix"]).reshape(3, 3).T
    depth_scale = 1000.0

intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, camera_matrix)
voxel_size = 0.05

color_filenames = sorted(glob.glob(f"{data_path}/color/*.jpg"))
depth_filenames = sorted(glob.glob(f"{data_path}/depth/*.png"))

prev = None
pose = np.eye(4)

i = 0
trajectory = []

for color_filename, depth_filename in zip(color_filenames, depth_filenames):
    color = cv2.imread(color_filename)
    depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        depth_scale=depth_scale,
        depth_trunc=3.86,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )

    # downsample
    pcd = pcd.voxel_down_sample(voxel_size)

    # compute normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

    # find features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    print(fpfh)

    if prev is None:
        prev = pcd, fpfh
        continue

    # i += 1
    # if i < 100:
    #     continue

    prev_pcd, prev_fpfh = prev
    prev = pcd, fpfh

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd, prev_pcd, fpfh, prev_fpfh, True,
        voxel_size*1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size*1.5)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    # result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    #     pcd, prev_pcd, fpfh, prev_fpfh,
    #     o3d.pipelines.registration.FastGlobalRegistrationOption(
    #         maximum_correspondence_distance=voxel_size*0.5))

    print(result.transformation)

    pose = pose @ result.transformation
    trajectory.append(pose[:3, 3])

    # draw_registration_result(pcd, prev_pcd, result.transformation)

    # break

    # visualize pcd and features
    # o3d.visualization.draw_geometries([pcd])

    # break




# plot trajectory
trajectory = np.array(trajectory)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


# pcl library 
# sudo apt install libboost-all-dev
# cmake -DCMAKE_BUILD_TYPE=Release ..
# make -j2
# sudo make -j2 install
