from dice_pose_estimator import DicePoseEstimator
import numpy as np
from pathlib import Path

data = np.load("realsense_data/dice_realsense_capture10.npz")
rgb = data["rgb"]
depth = data["depth"]
pcl = data["pcl"]

mesh_path = Path("meshes/untitled.obj")

est = DicePoseEstimator(mesh_path=mesh_path, verbose=True)

# 1) look at the model cloud alone
est.show(est.dice_cloud)

mask = est.binarize_pips(est.dice_cloud, visualize=True)

dice_clouds = est.segment_dice_pointcloud(pcl_xyz=pcl, pcl_rgb=rgb, visualize=True)

mask = est.binarize_pips(dice_clouds, visualize=True)

Ts_icp = est.align_icp(dice_clouds, visualize=False)

# 3) refine orientation using pips
Ts_final = est.refine_orientation_with_pips(
    dice_cloud=dice_clouds,
    T_icp=Ts_icp,
    match_radius=0.012,
    visualize=True,  # one window: real dice + oriented models
)

print("Final transforms:")
for i, T in enumerate(Ts_final):
    print(f"Die {i}:\n{T}")