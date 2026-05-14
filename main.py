import argparse
import robotic as ry
import numpy as np
import time
import cv2
from typing import Tuple, List, Any
from pprint import pprint
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import math
import matplotlib
matplotlib.use("TkAgg")

INPUT_NUM = 21

from dice_pose_estimator import DicePoseEstimator
from decision_module import (
    Action,
    candidate_first_actions_to_target_sum,
    plan_rai_main_primitives,
    sum_top_faces,
)

def seven_d_to_matrix(seven_d_pose):
    pos = seven_d_pose[:3]
    quat = seven_d_pose[3:] 
    quat_scipy = [quat[1], quat[2], quat[3], quat[0]]
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat_scipy).as_matrix()
    mat[:3, 3] = pos
    return mat

def matrix_to_7d(matrix):
    pos = matrix[:3, 3]
    rot_mat = matrix[:3, :3]
    quat_scipy = R.from_matrix(rot_mat).as_quat() 
    quat_rai = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
    return np.concatenate([pos, quat_rai])

def yaw_quat_from_rotmat(rot_mat: np.ndarray) -> np.ndarray:
    rot_mat = np.asarray(rot_mat, dtype=float).reshape(3, 3)
    yaw = float(math.atan2(rot_mat[1, 0], rot_mat[0, 0]))
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float)

def yaw_quat_from_pose7d(pose7d: np.ndarray) -> np.ndarray:
    pose7d = np.asarray(pose7d, dtype=float).reshape(7,)
    qw, qx, qy, qz = (float(x) for x in pose7d[3:7])
    rot_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return yaw_quat_from_rotmat(rot_mat)

def yaw_quat_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float)

def quat_x_up():
    s = math.sqrt(0.5)
    return np.array([s, 0.0, s, 0.0], dtype=float)

def quat_y_up():
    s = math.sqrt(0.5)
    return np.array([s, -s, 0.0, 0.0], dtype=float)

def quat_z_up():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

def rai_quat_from_scipy(rot: R):
    qx, qy, qz, qw = rot.as_quat()
    return np.array([qw, qx, qy, qz], dtype=float)


def gen_scene(
    filename: str = "configs/pandasTable_fixedCam.g",
) -> ry.Config:
    C = ry.Config()
    C.addFile(filename)

    box1_pos_init = [-0.05, 0.3, 0.68]
    box2_pos_init = [0.08, 0.28, 0.68]
    observation_pose = [-0.03, 0.1, 1.0, 0.258819, -0.965926, 0.0, 0.0]

    C.addFrame("box1").setPosition(box1_pos_init).setShape(
        ry.ST.ssBox, size=[0.06, 0.06, 0.06, 0.005]
    ).setColor([1.0, 1.0, 0.0]).setMass(0.1).setContact(1).setMeshFile("meshes/untitled.obj")

    C.addFrame("box2").setPosition(box2_pos_init).setShape(
        ry.ST.ssBox, size=[0.06, 0.06, 0.06, 0.005]
    ).setColor([0.0, 1.0, 1.0]).setMass(0.1).setContact(1).setMeshFile("meshes/untitled.obj")

    tip_x_up = R.from_euler('x', 90, degrees=True)
    tip_y_up = R.from_euler('x', 0, degrees=True)
    yaw = R.from_euler('z', 45, degrees=True) 

    q1 = rai_quat_from_scipy(yaw * tip_x_up)
    q2 = rai_quat_from_scipy(tip_y_up)

    C.getFrame("box1").setPose(np.concatenate([np.asarray(box1_pos_init, float), q1]))
    C.getFrame("box2").setPose(np.concatenate([np.asarray(box2_pos_init, float), q2]))

    C.addFrame("approachframe").setPose([0., 0., 1.0, 1., 0., 0., 0.]).setShape(type=ry.ST.marker, size=[0.03])

    C.addFile("./configs/markers.g")
    C.addFrame(name="observation_frame").setPose(observation_pose).setShape(type=ry.ST.marker, size=[0.05])
    C.addFrame(name="offset_obs_frame").setPose(observation_pose).setShape(type=ry.ST.marker, size=[0.02])

    cam = C.getFrame("l_cameraWrist")
    cam.setShape(type=ry.ST.marker, size=[0.05])

    C.addFrame("box1_axes", "box1").setShape(ry.ST.marker, size=[0.05])
    C.addFrame("box2_axes", "box2").setShape(ry.ST.marker, size=[0.05])

    return C

def set_approach_frame(C, goal_die, offset=0.1):
    parent_frame = C.getFrame(f"box{goal_die}_axes")
    pose = parent_frame.getPose().copy()
    pose[2] += offset
    C.getFrame("approachframe").setPose(pose)
    C.view()

def komo_obs_pos(komo: ry.KOMO, q_home: np.ndarray):
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q_home)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', 'observation_frame'], ry.OT.eq, [1e1])
    komo.addObjective([], ry.FS.scalarProductXX, ['l_gripper', 'observation_frame'], ry.OT.eq, [1e1], [-1])
    komo.addObjective([], ry.FS.scalarProductZZ, ['l_gripper', 'observation_frame'], ry.OT.eq, [1e1], [-1])
    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    return ret, komo.getPath()

def _komo_obs_offset(C: ry.Config, q_current: np.ndarray, target_frame: str) -> Tuple[ry.SolverReturn, np.ndarray]:
    # plans alignment of the gripper with target_frame (e.g. for lateral offsets)
    komo = ry.KOMO(config=C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=False)
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q_current)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', target_frame], ry.OT.eq, [1e1])
    komo.addObjective([], ry.FS.scalarProductXX, ['l_gripper', target_frame], ry.OT.eq, [1e1], [-1])
    komo.addObjective([], ry.FS.scalarProductZZ, ['l_gripper', target_frame], ry.OT.eq, [1e1], [-1])
    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    return ret, komo.getPath()

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0., 0., 1.])
    return v / n

def get_target_orientation_vectors(komo: ry.KOMO, die_frame: str, axis_raw: str):
    axis_char = axis_raw.lstrip('-')
    sign = -1.0 if axis_raw.startswith('-') else 1.0

    Rm = komo.getConfig().getFrame(die_frame).getRotationMatrix()
    col_idx = {'x': 0, 'y': 1, 'z': 2}[axis_char]

    axis_vec = np.array(Rm)[:, col_idx] * sign

    z_world = np.array([0., 0., 1.])
    axis_vec = axis_vec - np.dot(axis_vec, z_world) * z_world
    axis_vec = unit(axis_vec)

    if np.linalg.norm(axis_vec) < 1e-6:
        raise ValueError(f"Axis {axis_raw} is near world-z here; not a tabletop roll axis.")

    sideways_vec = unit(np.cross(axis_vec, z_world))
    z_pre = unit(z_world - sideways_vec)
    z_post = unit(z_world + sideways_vec)

    return axis_vec, z_pre, z_post


def _build_base_komo(C: ry.Config, q_home: np.ndarray, goal_die: str,
                     phases: int = 6, slices_per_phase: int = 1, k_order: int = 1,
                     enable_collisions: bool = True) -> ry.KOMO:
    komo = ry.KOMO(config=C, phases=phases, slicesPerPhase=slices_per_phase,
                   kOrder=k_order, enableCollisions=enable_collisions)

    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q_home)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    komo.addObjective([1.], ry.FS.positionDiff, ['l_gripper', 'approachframe'], ry.OT.eq, [1e1])
    komo.addObjective([2.], ry.FS.positionDiff, ['l_gripper', goal_die],        ry.OT.eq, [1e1])
    komo.addObjective([3., 4.], ry.FS.positionDiff, ['l_gripper', 'approachframe'], ry.OT.eq, [1e3])
    komo.addObjective([4., 5.], ry.FS.positionDiff, ['l_gripper', goal_die], ry.OT.eq,
                      np.eye(3) - np.outer([0, 0, 1], [0, 0, 1]))
    komo.addObjective([5.], ry.FS.positionDiff, ['l_gripper', goal_die],        ry.OT.eq, [1e1])
    komo.addObjective([6.], ry.FS.positionDiff, ['l_gripper', 'approachframe'], ry.OT.eq, [1e1])

    komo.addObjective([], ry.FS.distance, ['l_palm', goal_die], ry.OT.ineq, [1e1])
    return komo


def komo_ik(komo: ry.KOMO, q_home: np.ndarray, rot: tuple):
    axis_raw, die = rot
    goal_die = f'box{die}_axes'
    C = komo.getConfig()

    axis_vec, z_pre, z_post = get_target_orientation_vectors(komo, goal_die, axis_raw)

    # explore normal approach and opposite yaw approach
    branches = [
        ("A",  axis_vec,  z_pre,  z_post),
        ("B", -axis_vec,  z_pre,  z_post),
    ]

    best = None 

    for name, x_target, z_pre_b, z_post_b in branches:
        komo_b = _build_base_komo(C, q_home, goal_die,
                                  phases=6, slices_per_phase=1, k_order=1,
                                  enable_collisions=True)

        w_align = [1e1]
        komo_b.addObjective([1., 6.], ry.FS.vectorX, ['l_gripper'], ry.OT.eq, w_align, x_target)
        komo_b.addObjective([1., 3.], ry.FS.vectorZ, ['l_gripper'], ry.OT.eq, w_align, z_pre_b)
        komo_b.addObjective([4., 6.], ry.FS.vectorZ, ['l_gripper'], ry.OT.eq, w_align, z_post_b)

        ret = ry.NLP_Solver(komo_b.nlp(), verbose=0).solve()
        if not ret.feasible:
            continue

        q_path = komo_b.getPath()
        cost = motion_cost(q_path)

        if best is None or cost < best[0]:
            best = (cost, ret, q_path, name)

    if best is None:
        print(f"-- solution infeasible for axis {axis_raw}")
        return ret, None

    cost, ret, q_path, name = best
    print(f"[ik] chose branch {name} cost={cost:.4f} axis={axis_raw} die={die}")
    return ret, q_path

def botop_obs_pos(bot: ry.BotOp, C: ry.Config, q_obs: np.ndarray):
    bot.moveTo(q_obs[0])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

def botop_pickplace(bot: ry.BotOp, C: ry.Config, q_home: np.ndarray, q_pick: np.ndarray):
    bot.gripperMove(ry.ArgWord._left, .085, .1)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    bot.moveTo(q_pick[0])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_pick[1])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.gripperClose(ry.ArgWord._left, 10, 0.05, 0.1)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)
    time.sleep(1.0)
    
    while not bot.gripperDone(ry.ArgWord._left):
        bot.sync(C)

    bot.moveTo(q_pick[2])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_pick[3])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_pick[4])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.gripperMove(ry.ArgWord._left, .085, .1)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    bot.moveTo(q_pick[5])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_home)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)


def motion_cost(q_path: np.ndarray) -> float:
    q_path = np.asarray(q_path, dtype=float)
    if q_path.ndim != 2 or q_path.shape[0] < 2:
        return 0.0
    dq = np.diff(q_path, axis=0)
    return float(np.sum(np.linalg.norm(dq, axis=1)))

def check(C: ry.Config, message=""):
    if bool(getattr(args, "no_view", False)):
        return
    user_key = C.view(pause=True, message=f"{message} (Press q to quit)")
    if user_key == ord("q"):
        del C
        exit(0)

def check_estimate(C: ry.Config, q_home: np.ndarray, bot: ry.BotOp, message: str):
    LATERAL_OFFSET = 0.02 
    if bot is None:
        return
        
    obs_frame = C.getFrame("observation_frame")
    offset_frame = C.getFrame("offset_obs_frame")

    if bool(getattr(args, "no_view", False)):
        return

    user_key = C.view(pause=True, message=f"{message} (l=left 2cm, r=right 2cm, other=accept/reset, q=quit)")

    if user_key == ord("l"):
        pos = np.array(offset_frame.getPosition(), dtype=float)
        pos[0] -= LATERAL_OFFSET 
        offset_frame.setPosition(pos)
        offset_frame.setQuaternion(obs_frame.getQuaternion())

        q_now = C.getJointState().copy()
        ret, q_path = _komo_obs_offset(C, q_now, "offset_obs_frame")
        if ret.feasible and q_path is not None:
            bot.moveTo(q_path[0])
            bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
            time.sleep(0.5)
            estimate_dice_pose(C, bot, q_home)
        else:
            print("left move infeasible")
            return

    elif user_key == ord("r"):
        pos = np.array(offset_frame.getPosition(), dtype=float)
        pos[0] += LATERAL_OFFSET 
        offset_frame.setPosition(pos)
        offset_frame.setQuaternion(obs_frame.getQuaternion())

        q_now = C.getJointState().copy()
        ret, q_path = _komo_obs_offset(C, q_now, "offset_obs_frame")
        if ret.feasible and q_path is not None:
            bot.moveTo(q_path[0])
            bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
            time.sleep(0.5)
            estimate_dice_pose(C, bot, q_home)
        else:
            print("right move infeasible")
            return

    elif user_key == ord("q"):
        del C
        exit(0)

    else:
        offset_frame.setPosition(obs_frame.getPosition())
        offset_frame.setQuaternion(obs_frame.getQuaternion())
        return

def estimate_dice_pose(C, bot: ry.BotOp, q_home: np.ndarray):
    cameras = {"top": "cameraTopMiddle", "wrist": "l_cameraWrist"}
    cam_name = cameras[args.input]

    if bool(getattr(args, "skip_perception", False)):
        print("[pose] skipping perception; using existing scene poses")
        return

    estimator_pose_mode = str(getattr(args, "estimator_pose_mode", "pose"))
    if estimator_pose_mode not in ("pose", "position"):
        raise ValueError(f"Unknown --estimator_pose_mode {estimator_pose_mode!r}")

    # load inputs from disk rather than robot sensors if npz is provided
    if getattr(args, "npz", None):
        data = np.load(str(args.npz))
        rgb = data["rgb"]
        depth = data["depth"]
        pcl = data["pcl"]

        X_world_cam_7d = C.getFrame(cam_name).getPose()
        X_world_cam = seven_d_to_matrix(X_world_cam_7d)

        if args.display and (not bool(getattr(args, "no_display", False))):
            fig = plt.figure(figsize=(10, 5))
            axs = fig.subplots(1, 2)
            axs[0].imshow(rgb); axs[0].set_title("rgb")
            axs[1].matshow(depth); axs[1].set_title("depth")
            plt.show()

        quantile = 0.16 if args.real else 0.2
        solver = DicePoseEstimator(mesh_path="meshes/untitled.obj", quantile=quantile)
        poses_camera_frame = solver.estimate(pcl_xyz=pcl, pcl_rgb=rgb, visualize=not bool(getattr(args, "no_display", False)))

        if poses_camera_frame:
            T_world_dice = X_world_cam @ poses_camera_frame
            pose_dice1 = matrix_to_7d(T_world_dice[0])
            pose_dice2 = matrix_to_7d(T_world_dice[1])

            if estimator_pose_mode == "pose":
                C.getFrame("box1").setPose(pose_dice1)
                C.getFrame("box2").setPose(pose_dice2)
            else:
                unit_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                C.getFrame("box1").setPose(np.concatenate([pose_dice1[:3], unit_quat]))
                C.getFrame("box2").setPose(np.concatenate([pose_dice2[:3], unit_quat]))

            if not bool(getattr(args, "no_view", False)):
                C.view()
            print("updated frames to detected pose.")
        else:
            print("no dice found in npz capture.")
        return

    # standard capture and estimate loop
    if args.display and args.input is not None:
        poses_camera_frame = []
        num_tries = int(getattr(args, "perception_max_tries", 3))

        for t in range(num_tries):
            rgb, depth, pcl = bot.getImageDepthPcl(cam_name)

            X_world_cam_7d = C.getFrame(cam_name).getPose()
            X_world_cam = seven_d_to_matrix(X_world_cam_7d)

            fig = plt.figure(figsize=(10, 5))
            axs = fig.subplots(1, 2)
            axs[0].imshow(rgb)
            axs[1].matshow(depth)
            plt.show()

            quantile = 0.16 if args.real else 0.2
            solver = DicePoseEstimator(mesh_path="meshes/untitled.obj", quantile=quantile)

            poses_camera_frame = solver.estimate(pcl_xyz=pcl, pcl_rgb=rgb, visualize=True)
            if len(poses_camera_frame) == 2:
                print(f"\nsuccess! found {len(poses_camera_frame)} dice.")
                break
            print(f"\nfound only {len(poses_camera_frame)} dice. retrying...")

        if len(poses_camera_frame) == 2:
            T_world_dice = X_world_cam @ poses_camera_frame

            pose_dice1 = matrix_to_7d(T_world_dice[0])
            pose_dice2 = matrix_to_7d(T_world_dice[1])

            # pose_dice1[0] -= 0.01 # this is hardcoded for real robot, maybe camera was not correct calibrated
            # pose_dice1[1] -= 0.01 # this is hardcoded for real robot, maybe camera was not correct calibrated

            if estimator_pose_mode == "pose":
                C.getFrame("box1").setPose(pose_dice1)
                C.getFrame("box2").setPose(pose_dice2)
            else:
                unit_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                C.getFrame("box1").setPose(np.concatenate([pose_dice1[:3], unit_quat]))
                C.getFrame("box2").setPose(np.concatenate([pose_dice2[:3], unit_quat]))

            C.view()
            print("updated frames to detected pose.")
            
        else:
            print(f"no dice found after {num_tries} tries; switching to --skip_perception.")
            args.skip_perception = True

        check_estimate(C, q_home, bot, message="")
        bot.moveTo(q_home)
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)


def move_to_obs_pos(C, bot, q_home):
    komo = ry.KOMO(config=C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=False)
    ret_obs, q_obs = komo_obs_pos(komo=komo, q_home=q_home)
    
    global INPUT_NUM
    INPUT_NUM += 1
    del komo

    if ret_obs.feasible:
        botop_obs_pos(bot=bot, C=C, q_obs=q_obs)
        check(C, "reached observation pose")
    else:
        print("observation pose infeasible")
        exit(0)


def main(args):
    ry.params_add({
        "physx/angularDamping": 0.1,
        "physx/defaultFriction": 100.0,
        "physx/defaultRestitution": 0.7,
        "botsim/verbose": 0
    })

    C = gen_scene()

    check(C, "scene loaded")
    q_home = C.getJointState().copy()
    bot = None
    if args.npz is None:
        bot = ry.BotOp(C=C, useRealRobot=args.real)

    move_to_obs_pos(C, bot, q_home)
    estimate_dice_pose(C, bot, q_home)

    def rot_by_box(box_id: int) -> np.ndarray:
        return np.array(C.getFrame(f"box{int(box_id)}_axes").getRotationMatrix(), dtype=float).reshape(3, 3)

    boxes = (1, 2)

    if args.target_sum is not None:
        target_sum = int(args.target_sum)
        max_steps = int(args.decision_max_steps)

        if args.npz is not None:
            start_R_by_box = {b: rot_by_box(b) for b in boxes}
            plan = plan_rai_main_primitives(
                start_R_by_box=start_R_by_box,
                target_sum=target_sum,
                boxes=boxes,
                max_steps=max_steps,
            )
            print(f"[offline] plan={plan}")
            if plan:
                rot = plan[0] 
                q_home = C.getJointState().copy()
                set_approach_frame(C, rot[1])
                
                komo = ry.KOMO(config=C, phases=6, slicesPerPhase=1, kOrder=1, enableCollisions=True)
                ret_pick, q_pick = komo_ik(komo=komo, q_home=q_home, rot=rot)
                del komo
                print(f"[offline] first_step_ik_feasible={bool(ret_pick.feasible)} primitive={rot}")
                
            if bot is not None:
                del bot
            del C
            return

        for step_idx in range(max_steps):
            cur_sum = sum_top_faces([rot_by_box(b) for b in boxes])
            print(f"[decision] step={step_idx}/{max_steps} current_sum={cur_sum} target_sum={target_sum}")
            if cur_sum == target_sum:
                print("[decision] target reached")
                break

            remaining = max_steps - step_idx
            start_R_by_box = {b: rot_by_box(b) for b in boxes}

            if args.decision_cost_mode == "steps":
                plan = plan_rai_main_primitives(
                    start_R_by_box=start_R_by_box,
                    target_sum=target_sum,
                    boxes=boxes,
                    max_steps=remaining,
                )
                if not plan:
                    print("[decision] no plan found for target sum")
                    break
                rot = plan[0]
                print(f"[decision] executing primitive={rot} (remaining_plan_len={len(plan)})")

            elif args.decision_cost_mode == "motion":
                candidates: List[Action] = candidate_first_actions_to_target_sum(
                    start_R_by_box=start_R_by_box,
                    target_sum=target_sum,
                    boxes=boxes,
                    max_steps=remaining,
                )
                if not candidates:
                    print("[decision] no plan found for target sum")
                    break

                candidate_prims: List[Tuple[str, int]] = []
                for box_id, axis, angle in candidates:
                    axis_raw = f"-{axis}" if angle < 0 else axis
                    candidate_prims.append((axis_raw, int(box_id)))

                best: tuple[float, tuple[str, int], np.ndarray] | None = None
                for cand in candidate_prims:
                    q_home = C.getJointState().copy()
                    set_approach_frame(C, cand[1])

                    komo = ry.KOMO(config=C, phases=6, slicesPerPhase=1, kOrder=1, enableCollisions=True)
                    ret_pick, q_pick = komo_ik(komo=komo, q_home=q_home, rot=cand)
                    del komo
                    if not ret_pick.feasible or q_pick is None:
                        continue

                    cost = motion_cost(q_pick)
                    if best is None or cost < best[0]:
                        best = (cost, cand, q_pick)

                if best is None:
                    print("[decision] all candidate primitives infeasible in ik")
                    break

                best_cost, rot, _ = best
                print(f"[decision] selected primitive={rot} motion_cost={best_cost:.3f} (candidates={len(candidate_prims)})")

            else:
                raise ValueError(f"Unknown decision_cost_mode: {args.decision_cost_mode}")

            q_home = C.getJointState().copy()
            set_approach_frame(C, rot[1])

            komo = ry.KOMO(config=C, phases=6, slicesPerPhase=1, kOrder=1, enableCollisions=True)
            ret_pick, q_pick = komo_ik(komo=komo, q_home=q_home, rot=rot)

            if ret_pick.feasible:
                check(C, "ik solved. ready to pick?")
                if not bool(getattr(args, "dry_run", False)):
                    if bot is None:
                        raise RuntimeError("execution requires a BotOp instance.")
                    botop_pickplace(bot=bot, C=C, q_home=q_home, q_pick=q_pick)
                    check(C, "done")
                else:
                    print("[decision] ik feasible (dry-run; pass --dry_run=false to move)")
            else:
                print("pick ik infeasible")

            del komo

            if args.npz is None and (not bool(getattr(args, "dry_run", False))):
                move_to_obs_pos(C, bot, q_home)
                estimate_dice_pose(C, bot, q_home)

    else:
        decision = [('y', 2), ('-z', 2), ('-x', 2), ('y', 1), ('-z', 1), ('-x', 1)]
        for rot in decision:
            q_home = C.getJointState().copy()
            set_approach_frame(C, rot[1])

            komo = ry.KOMO(config=C, phases=6, slicesPerPhase=1, kOrder=1, enableCollisions=True)
            ret_pick, q_pick = komo_ik(komo=komo, q_home=q_home, rot=rot)

            if ret_pick.feasible:
                check(C, "ik solved. ready to pick?")
                if not bool(getattr(args, "dry_run", False)):
                    if bot is None:
                        raise RuntimeError("execution requires a BotOp instance.")
                    botop_pickplace(bot=bot, C=C, q_home=q_home, q_pick=q_pick)
                    check(C, "done")
                else:
                    print("[decision] ik feasible (dry-run)")
            else:
                print("pick ik infeasible")

            del komo
            if args.npz is None and (not bool(getattr(args, "dry_run", False))):
                move_to_obs_pos(C, bot, q_home)
                estimate_dice_pose(C, bot, q_home)

    if bot is not None:
        del bot
    del C


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", default=False, help="Use real robot")
    p.add_argument("--input", type=str, default="wrist", help="cameraWrist or cameraTopMiddle")
    p.add_argument("--display", action="store_true", default=True, help="Show camera view")
    p.add_argument("--no_display", action="store_true", default=False, help="Disable matplotlib/Open3D visualization")
    p.add_argument("--no_view", action="store_true", default=False, help="Disable RAI viewer popups/pauses")
    p.add_argument("--npz", type=str, default=None, help="Offline mode: load rgb/depth/pcl from a saved .npz capture")
    p.add_argument("--skip_perception", action="store_true", default=False)
    p.add_argument("--estimator_pose_mode", type=str, default="pose", choices=["pose", "position"])
    p.add_argument("--perception_max_tries", type=int, default=3)
    p.add_argument("--dry_run", action="store_true", default=False)
    p.add_argument("--target_sum", type=int, default=7)
    p.add_argument("--decision_max_steps", type=int, default=8)
    p.add_argument("--decision_cost_mode", type=str, default="steps", choices=["steps", "motion"])
    args = p.parse_args()
    main(args)