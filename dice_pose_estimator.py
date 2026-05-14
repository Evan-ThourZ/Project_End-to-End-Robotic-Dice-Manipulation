import open3d as o3d
import numpy as np
import copy

class DicePoseEstimator:
    def __init__(self, mesh_path, quantile=0.16, verbose=False):
        self.quantile = quantile
        self.verbose = verbose
        self.plane_distance_threshold = 0.007
        self.axis_size = 0.1
        
        self.dice_mesh, self.dice_cloud = self.load_dice_model(mesh_path)
        self.model_pip_mask = self.binarize_pips(self.dice_cloud)
        self.model_pip_indices = np.where(self.model_pip_mask)[0]
        self.model_body_indices = np.where(~self.model_pip_mask)[0]
        self.cube_rotations = self.generate_cube_rotations()

    def generate_cube_rotations(self):
        axes = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1]),
        ]
        rots = []
        for up in axes:
            for fwd in axes:
                if np.abs(np.dot(up, fwd)) < 1e-6:
                    right = np.cross(fwd, up)
                    R = np.column_stack((right, fwd, up))
                    if np.linalg.det(R) > 0.5:
                        rots.append(R)
        return rots

    def load_dice_model(self, mesh_path):
        mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
        mesh.compute_vertex_normals()
        model_cloud = mesh.sample_points_uniformly(number_of_points=15000)
        model_cloud.estimate_normals()
        
        center = model_cloud.get_center()
        model_cloud.translate(-center)
        mesh.translate(-center)
        
        return mesh, model_cloud

    def segment_dice_pointcloud(self, pcl_xyz, pcl_rgb, visualize=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl_xyz.reshape(-1, 3).astype(float))
        pcd.colors = o3d.utility.Vector3dVector(pcl_rgb.reshape(-1, 3).astype(float) / 255.0)
        pcd.estimate_normals()

        # drop the table plane and points outside valid z-range
        _, inliers = pcd.segment_plane(self.plane_distance_threshold, 3, 1000)
        non_table = pcd.select_by_index(inliers, invert=True)

        pts = np.asarray(non_table.points)
        keep = (pts[:, 2] > 0.0) & (pts[:, 2] < 1.2)
        non_table = non_table.select_by_index(np.where(keep)[0].tolist())

        labels = np.array(non_table.cluster_dbscan(eps=0.03, min_points=120, print_progress=False))
        if labels.size == 0 or labels.max() < 0: 
            return []

        candidates = []
        for i in range(labels.max() + 1):
            cluster = non_table.select_by_index(np.where(labels == i)[0])
            cluster, _ = cluster.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            # filter by physical bounding box size
            if 0.03 < np.max(cluster.get_axis_aligned_bounding_box().get_extent()) < 0.10:
                candidates.append(cluster)

        dice_clouds = sorted(candidates, key=lambda c: len(c.points), reverse=True)[:2]

        if visualize: 
            self.show(dice_clouds)
        return dice_clouds

    def binarize_pips(self, cloud, visualize=False):
        # returns boolean mask: True = pip (dark), False = body (light)
        is_list = isinstance(cloud, (list, tuple))
        clouds = list(cloud) if is_list else [cloud]

        masks = []
        geoms = []

        for c in clouds:
            colors = np.asarray(c.colors)
            if len(colors) == 0:
                masks.append(np.array([], dtype=bool))
                continue

            gray = colors.mean(axis=1)
            thr = np.quantile(gray, self.quantile)
            mask = gray < thr
            masks.append(mask)

            if visualize:
                pip = c.select_by_index(np.where(mask)[0])
                body = c.select_by_index(np.where(~mask)[0])
                pip.paint_uniform_color([1, 0, 0])
                body.paint_uniform_color([0.8, 0.8, 0.8])
                geoms.extend([body, pip])

        if visualize and geoms:
            self.show(geoms)

        return masks if is_list else masks[0]

    def align_icp(self, cloud, max_iter=60, visualize=False):
        is_list = isinstance(cloud, (list, tuple))
        clouds = list(cloud) if is_list else [cloud]

        voxel_size = 0.005
        coarse_threshold = 0.020
        fine_threshold = 0.005

        model = copy.deepcopy(self.dice_cloud)
        model_ds = model.voxel_down_sample(voxel_size)
        model_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

        loss = o3d.pipelines.registration.HuberLoss(k=0.01)
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        transforms = []
        vis_geoms = []

        for target_raw in clouds:
            target_ds = target_raw.voxel_down_sample(voxel_size)
            target_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

            # center target prior to icp to prevent floating
            target_center = target_ds.get_axis_aligned_bounding_box().get_center()
            target_centered = copy.deepcopy(target_ds).translate(-target_center)

            reg1 = o3d.pipelines.registration.registration_icp(
                model_ds, target_centered, coarse_threshold, np.eye(4), p2p,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15)
            )

            reg2 = o3d.pipelines.registration.registration_icp(
                model_ds, target_centered, fine_threshold, reg1.transformation, p2p,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
            )

            # restore world coordinates
            T_final = np.eye(4)
            T_final[:3, 3] = target_center
            T_final = T_final @ reg2.transformation

            transforms.append(T_final)

            if visualize:
                vis_mod = copy.deepcopy(self.dice_cloud).transform(T_final)
                vis_mod.paint_uniform_color([0, 1, 0])
                vis_geoms.extend([target_raw, vis_mod])

        if visualize and vis_geoms:
            self.show(vis_geoms)

        return transforms if is_list else transforms[0]

    def refine_orientation_with_pips(self, dice_cloud, T_icp, match_radius=0.001, visualize=False):
        is_list = isinstance(dice_cloud, (list, tuple))
        clouds = list(dice_cloud) if is_list else [dice_cloud]
        Ts_icp = list(T_icp) if is_list else [T_icp]

        model_pts = np.asarray(self.dice_cloud.points)
        model_pip_idx = self.model_pip_indices   

        T_best_all = []
        vis_geoms = []

        for cloud, T0 in zip(clouds, Ts_icp):
            real_mask = self.binarize_pips(cloud)

            if real_mask.sum() == 0:
                T_best_all.append(T0)
                continue

            best_T = T0
            best_score = -1e9
            
            cloud_normals = copy.deepcopy(cloud)
            if not cloud_normals.has_normals(): 
                cloud_normals.estimate_normals()

            for R in self.cube_rotations:
                R4 = np.eye(4)
                R4[:3, :3] = R
                T_candidate = T0 @ R4

                # quick icp to snap the current rotation guess and fix minor drift
                reg_snap = o3d.pipelines.registration.registration_icp(
                    self.dice_cloud, cloud_normals, 0.006, T_candidate,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8)
                )
                T_snapped = reg_snap.transformation

                score = self._pip_orientation_score(
                    T_snapped,
                    model_pts,
                    model_pip_idx,
                    real_cloud=cloud,
                    real_pip_mask=real_mask,
                    radius=match_radius,
                    tp_weight=2.0,
                    fp_weight=3.0,
                    fn_weight=1.5,
                )

                if score > best_score:
                    best_score = score
                    best_T = T_snapped

            T_best_all.append(best_T)

            if visualize:
                real_vis = copy.deepcopy(cloud)
                cols = np.asarray(real_vis.colors)
                cols[real_mask] = [1, 0, 0]
                real_vis.colors = o3d.utility.Vector3dVector(cols)

                model_vis = copy.deepcopy(self.dice_cloud)
                model_vis.transform(best_T)

                vis_geoms.extend([real_vis, model_vis])

        if visualize and vis_geoms:
            self.show(vis_geoms)

        return T_best_all if is_list else T_best_all[0]

    def _pip_orientation_score(
        self, T, model_pts, model_pip_idx, real_cloud, real_pip_mask,
        radius, tp_weight=2.0, fp_weight=3.0, fn_weight=1.5
    ):
        model_pips = model_pts[model_pip_idx]
        if len(model_pips) == 0:
            return -1e9

        # transform model pip points into camera frame
        pips_h = np.hstack([model_pips, np.ones((len(model_pips), 1))])
        pips_cam = (T @ pips_h.T).T[:, :3]

        real_pts = np.asarray(real_cloud.points)
        real_pip_pts = real_pts[real_pip_mask]
        N_real = len(real_pip_pts)

        # all real points (for FP check)
        real_tree_all = o3d.geometry.KDTreeFlann(real_cloud)

        # only real pip points (for TP check)
        pip_pcd = o3d.geometry.PointCloud()
        pip_pcd.points = o3d.utility.Vector3dVector(real_pip_pts)
        pip_tree = o3d.geometry.KDTreeFlann(pip_pcd)

        # transformed model pips (for FN check)
        model_pcd = o3d.geometry.PointCloud()
        model_pcd.points = o3d.utility.Vector3dVector(pips_cam)
        model_tree = o3d.geometry.KDTreeFlann(model_pcd)

        TP = 0
        FP = 0
        for p in pips_cam:
            k, _, _ = pip_tree.search_hybrid_vector_3d(p, radius, 1)
            if k > 0:
                TP += 1
                continue

            k2, _, _ = real_tree_all.search_hybrid_vector_3d(p, radius, 1)
            if k2 > 0:
                FP += 1

        matched_real = 0
        for rp in real_pip_pts:
            k, _, _ = model_tree.search_hybrid_vector_3d(rp, radius, 1)
            if k > 0:
                matched_real += 1
        FN = N_real - matched_real

        denom = (tp_weight * TP + fp_weight * FP + fn_weight * FN) + 1e-6
        return TP / denom

    def sort_transforms_by_camera_x(self, Ts):
        if Ts is None: return []
        Ts = list(Ts)
        if len(Ts) <= 1: return Ts
        return sorted(Ts, key=lambda T: float(T[0, 3]))

    def estimate(self, pcl_xyz, pcl_rgb, match_radius=0.001, visualize=False):
        dice_clouds = self.segment_dice_pointcloud(pcl_xyz, pcl_rgb, visualize=False)
        if not dice_clouds: 
            return []
        
        Ts_icp = self.align_icp(dice_clouds, visualize=False)
        Ts_final = self.refine_orientation_with_pips(
            dice_clouds, Ts_icp, match_radius=match_radius, visualize=visualize
        )

        Ts_final = self.sort_transforms_by_camera_x(Ts_final)

        if self.verbose and len(Ts_final) == 2:
            print(f"[estimate] cam-x: die0={Ts_final[0][0,3]:.4f}, die1={Ts_final[1][0,3]:.4f}")

        return Ts_final

    def show(self, geometry):
        geoms = list(geometry) if isinstance(geometry, (list, tuple)) else [geometry]
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_size))
        o3d.visualization.draw_geometries(geoms)