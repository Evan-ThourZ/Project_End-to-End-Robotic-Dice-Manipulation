import open3d as o3d
import numpy as np
import copy

class DicePoseEstimator:
    def __init__(self, mesh_path, quantile = 0.12, verbose=False):
        self.quantile = quantile
        self.verbose = verbose
        self.plane_distance_threshold = 0.009
        self.axis_size = 0.1
        self.dice_mesh, self.dice_cloud = self.load_dice_model(mesh_path)
        self.model_pip_mask = self.binarize_pips(self.dice_cloud)
        self.model_pip_indices = np.where(self.model_pip_mask)[0]
        self.model_body_indices = np.where(~self.model_pip_mask)[0]
        self.cube_rotations = self.generate_cube_rotations()

    def generate_cube_rotations(self):
        axes = [
            np.array([1,0,0]), np.array([-1,0,0]),
            np.array([0,1,0]), np.array([0,-1,0]),
            np.array([0,0,1]), np.array([0,0,-1]),
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
        return mesh, model_cloud

    def segment_dice_pointcloud(self, pcl_xyz, pcl_rgb, visualize=False):
        pcl_xyz = pcl_xyz.reshape(-1, 3)
        pcl_rgb = pcl_rgb.reshape(-1, 3)
        rgb = pcl_rgb.astype(float) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl_xyz.astype(float))
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)) #The two key arguments radius = 0.1 and max_nn = 30 specifies search radius and maximum nearest neighbor. It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.plane_distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        non_table = pcd.select_by_index(inliers, invert=True)

        pts = np.asarray(non_table.points)
        keep = (pts[:, 2] > 0.0) & (pts[:, 2] < 1.2)   # <-- pick a sensible max depth
        non_table = non_table.select_by_index(np.where(keep)[0].tolist())

        labels = np.array(non_table.cluster_dbscan(
            eps=0.03, min_points=100, print_progress=False
        ))

        if labels.size == 0 or np.all(labels == -1):
            return [non_table]

        unique = [c for c in np.unique(labels) if c != -1]
        dice_clouds = []
        for cid in unique:
            idx = np.where(labels == cid)[0]
            cloud = non_table.select_by_index(idx.tolist())
            dice_clouds.append(cloud)

        if visualize:
            self.show(dice_clouds)
        return dice_clouds

    def binarize_pips(self, cloud, visualize=False):
        """ Returns a boolean mask: True = Pip (dark), False = Body (light) """
        is_list = isinstance(cloud, (list, tuple))
        clouds = list(cloud) if is_list else [cloud]

        masks = []
        geoms = []

        for c in clouds:
            colors = np.asarray(c.colors)
            if len(colors) == 0:
                mask = np.array([], dtype=bool)
                masks.append(mask)
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

    def align_icp(self, cloud, max_iter=100, visualize=False):
        is_list = isinstance(cloud, (list, tuple))
        clouds = list(cloud) if is_list else [cloud]
        transforms = []
        vis_geoms = []

        for c in clouds:
            target = copy.deepcopy(c)
            source = copy.deepcopy(self.dice_cloud)

            src_center = source.get_center()
            tgt_center = target.get_center()
            init = np.eye(4)
            init[:3, 3] = tgt_center - src_center

            reg = o3d.pipelines.registration.registration_icp(
                source, target, 0.005, init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
            )
            transforms.append(reg.transformation)
            
            if visualize:
                src_vis = copy.deepcopy(source).transform(reg.transformation)
                src_vis.paint_uniform_color([0, 0, 1])
                vis_geoms.extend([target, src_vis])

        if visualize and vis_geoms:
            self.show(vis_geoms)

        return transforms if is_list else transforms[0]

    def refine_orientation_with_pips(self, dice_cloud, T_icp, match_radius=0.008, visualize=False):
        is_list = isinstance(dice_cloud, (list, tuple))
        clouds = list(dice_cloud) if is_list else [dice_cloud]
        Ts_icp = list(T_icp) if is_list else [T_icp]

        model_pts = np.asarray(self.dice_cloud.points)
        model_pip_idx = self.model_pip_indices   # computed once in __init__

        T_best_all = []
        vis_geoms = []

        for cloud, T0 in zip(clouds, Ts_icp):

            real_mask = self.binarize_pips(cloud)

            if real_mask.sum() == 0:
                T_best_all.append(T0)
                continue

            best_T = T0
            best_score = -1e9

            for R in self.cube_rotations:
                R4 = np.eye(4)
                R4[:3, :3] = R
                T_candidate = T0 @ R4

                score = self._pip_orientation_score(
                    T_candidate,
                    model_pts,
                    model_pip_idx,
                    real_cloud=cloud,
                    real_pip_mask=real_mask,
                    radius=match_radius,
                    fp_weight=2.0,
                    use_f1=True,
                )

                if score > best_score:
                    best_score = score
                    best_T = T_candidate

            T_best_all.append(best_T)

            if visualize:
                import copy
                # Show real pips as red
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
        self,
        T,
        model_pts,
        model_pip_idx,
        real_cloud,
        real_pip_mask,
        radius,
        fp_weight=2.0,
        use_f1=True,
    ):

        model_pips = model_pts[model_pip_idx]
        N_model = len(model_pips)
        if N_model == 0:
            return -1e9

        # --- transform model pip points into camera frame
        pips_h = np.hstack([model_pips, np.ones((N_model, 1))])
        pips_cam = (T @ pips_h.T).T[:, :3]

        # --- real pip points
        real_pts = np.asarray(real_cloud.points)
        real_pip_pts = real_pts[real_pip_mask]
        N_real = len(real_pip_pts)

        # KDTree for real cloud (for model->real queries)
        real_tree = o3d.geometry.KDTreeFlann(real_cloud)

        # KDTree for transformed model pips (for real->model queries)
        model_pcd = o3d.geometry.PointCloud()
        model_pcd.points = o3d.utility.Vector3dVector(pips_cam)
        model_tree = o3d.geometry.KDTreeFlann(model_pcd)

        # --- model -> real : count TP and FP (contradictions)
        TP = 0
        FP = 0
        for p in pips_cam:
            k, idxs, _ = real_tree.search_hybrid_vector_3d(p, radius, 1)
            if k == 0:
                continue
            idx = idxs[0]
            if real_pip_mask[idx]:
                TP += 1
            else:
                FP += 1

        # --- real -> model : count FN (unexplained real pips)
        # (equivalently count matched real pips; FN = N_real - matched)
        matched_real = 0
        for rp in real_pip_pts:
            k, _, _ = model_tree.search_hybrid_vector_3d(rp, radius, 1)
            if k > 0:
                matched_real += 1
        FN = N_real - matched_real

        # --- normalize into a single score
        if use_f1:
            denom = (2 * TP + fp_weight * FP + FN) + 1e-6
            return (2 * TP) / denom
        else:
            # weighted sum alternative (sometimes easier to tune)
            prec = TP / (N_model + 1e-6)
            rec  = matched_real / (N_real + 1e-6) if N_real > 0 else 0.0
            fp   = FP / (N_model + 1e-6)
            return prec + rec - fp_weight * fp



    def estimate(self, pcl_xyz, pcl_rgb, match_radius=0.01, visualize=False):
        dice_clouds = self.segment_dice_pointcloud(pcl_xyz, pcl_rgb, visualize=False)
        if not dice_clouds: return []
        
        Ts_icp = self.align_icp(dice_clouds, visualize=False)
        Ts_final = self.refine_orientation_with_pips(
            dice_clouds, Ts_icp, match_radius=match_radius, visualize=visualize
        )
        return Ts_final

    def show(self, geometry):
        geoms = list(geometry) if isinstance(geometry, (list, tuple)) else [geometry]
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_size))
        o3d.visualization.draw_geometries(geoms)