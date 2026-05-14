from dice_pose_estimator import DicePoseEstimator
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

data = np.load("dice_realsense_capture7.npz") #7,8,9,12,15,17,18,19,20,24
rgb = data["rgb"]
depth = data["depth"]
pcl = data["pcl"]

fig = plt.figure(figsize=(10,5))
axs = fig.subplots(1, 2)
axs[0].imshow(rgb)
axs[1].matshow(depth)    
plt.show()

mesh_path = Path("meshes/untitled.obj")

est = DicePoseEstimator(mesh_path=mesh_path, verbose=True)

est.show(est.dice_cloud)

mask = est.binarize_pips(est.dice_cloud, visualize=True)

dice_clouds = est.segment_dice_pointcloud(pcl_xyz=pcl, pcl_rgb=rgb, visualize=True)
print(len(dice_clouds))

mask = est.binarize_pips(dice_clouds, visualize=True)

Ts_icp = est.align_icp(dice_clouds, visualize=False)

Ts_final = est.refine_orientation_with_pips(
    dice_cloud=dice_clouds,
    T_icp=Ts_icp,
    match_radius=0.001,
    visualize=True,
)

print("Final transforms:")
for i, T in enumerate(Ts_final):
    print(f"Die {i}:\n{T}")