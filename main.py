import argparse
import robotic as ry
import numpy as np
import cv2
from typing import Tuple, List, Any
from pprint import pprint
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import math
import matplotlib
matplotlib.use("TkAgg")

# Import your solver (Ensure the file is named segmentation.py or dice_solver.py)
from dice_pose_estimator import DicePoseEstimator 

# --- HELPER FUNCTIONS FOR MATRIX MATH ---
def seven_d_to_matrix(seven_d_pose):
    """Converts [x,y,z, qw,qx,qy,qz] to 4x4 numpy matrix"""
    pos = seven_d_pose[:3]
    quat = seven_d_pose[3:] # [qw, qx, qy, qz]
    
    # Scipy Rotation expects [qx, qy, qz, qw] (scalar last)
    # Rai uses [qw, qx, qy, qz] (scalar first)
    quat_scipy = [quat[1], quat[2], quat[3], quat[0]]
    
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat_scipy).as_matrix()
    mat[:3, 3] = pos
    return mat

def matrix_to_7d(matrix):
    """Converts 4x4 numpy matrix to [x,y,z, qw,qx,qy,qz]"""
    pos = matrix[:3, 3]
    rot_mat = matrix[:3, :3]
    
    # Get quat [qx, qy, qz, qw]
    quat_scipy = R.from_matrix(rot_mat).as_quat()
    
    # Convert to Rai format [qw, qx, qy, qz]
    quat_rai = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
    
    return np.concatenate([pos, quat_rai])

# --- SCENE GENERATION ---
def gen_scene(filename: str = "configs/pandasTable_fixedCam.g") -> ry.Config:
    C = ry.Config()
    C.addFile(filename)
    
    # Initial Setup
    #base1_pos_init = [-0.25, 0.2, 0.6625]
    #base2_pos_init = [0.25, 0.2, 0.6625]
    box1_pos_init = [-0.15, 0.35, 0.68]
    box2_pos_init = [0.15, 0.35, 0.68]

    observation_pose = [
        -0.00, 0.1, 1.1,
        0.258819, -0.965926, 0.0, 0.0
    ]

    #C.addFrame("base1").setPosition(base1_pos_init).setShape(ry.ST.ssBox, size=[0.08, 0.08, 0.025, 0.005]).setColor([0.37, 0.37, 0.37]).setContact(1)
    C.addFrame("box1").setPosition(box1_pos_init).setShape(ry.ST.ssBox, size=[0.06, 0.06, 0.06, 0.005]).setColor([1.0, 1.0, 0.0]).setMass(0.1).setContact(1).setMeshFile("meshes/untitled.obj")

    #C.addFrame("base2").setPosition(base2_pos_init).setShape(ry.ST.ssBox, size=[0.08, 0.08, 0.025, 0.005]).setColor([0.37, 0.37, 0.37]).setContact(1)
    C.addFrame("box2").setPosition(box2_pos_init).setShape(ry.ST.ssBox, size=[0.06, 0.06, 0.06, 0.005]).setColor([0.0, 1.0, 1.0]).setMass(0.1).setContact(1).setMeshFile("meshes/untitled.obj")

    # Helper Frames
    C.addFrame(name="box1_axes").setPosition(box1_pos_init).setShape(type=ry.ST.marker, size=[0.05])
    C.addFrame(name="box2_axes").setPosition(box2_pos_init).setShape(type=ry.ST.marker, size=[0.05])
    C.addFrame("approachframe").setPose([0., 0., 1.0, 1., 0., 0., 0.]).setShape(type=ry.ST.marker, size=[0.03])

    C.addFile("./configs/markers.g")
    C.addFrame(name="observation_frame").setPose(observation_pose).setShape(type=ry.ST.marker, size=[0.05])

    cam = C.getFrame("l_cameraWrist")
    cam.setShape(type=ry.ST.marker, size=[0.05])


    return C

def set_approach_frame(C, goal_die, offset=0.12):
    parent_frame = C.getFrame(f"box{goal_die}")
    pose = parent_frame.getPose()
    pos = pose[:3]
    new_pos = pos.copy()
    new_pos[2] += offset

    approachframe = C.getFrame("approachframe")

    new_pose = np.concatenate([new_pos, np.array([1., 0., 0., 0.])])
    approachframe.setPose(new_pose)

    C.view()


def komo_obs_pos(komo: ry.KOMO, q_home: np.ndarray) -> Tuple[ry.SolverReturn, ry.KOMO, np.ndarray]:
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q_home)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', 'observation_frame'], ry.OT.eq, [1e1])
    # Orientation constraints
    komo.addObjective([], ry.FS.scalarProductXX, ['l_gripper', 'observation_frame'], ry.OT.eq, [1e1], [-1])
    komo.addObjective([], ry.FS.scalarProductZZ, ['l_gripper', 'observation_frame'], ry.OT.eq, [1e1], [-1])
    
    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    return ret, komo.getPath()

def komo_ik(
    komo: ry.KOMO,
    q_home: np.ndarray,
    rot: tuple,
) -> Tuple[ry.SolverReturn, np.ndarray]:

    axis_raw, die = rot
    goal_die = f'box{die}_axes'
    tilt = math.cos(math.radians(45.0))   # ≈ 0.707

    # interpret axis: 'x', '-x', 'y', '-y'
    axis_sign = 1.0
    if isinstance(axis_raw, str) and axis_raw.startswith('-'):
        axis = axis_raw[1:]   # 'x' or 'y'
        axis_sign = -1.0
    else:
        axis = axis_raw

    if axis not in ('x', 'y'):
        raise ValueError(f"axis must be 'x', '-x', 'y' or '-y', got {axis_raw}")

    # rotation direction: +1 for 'x','y'; -1 for '-x','-y'
    rot_sign = axis_sign

    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q_home)
    #komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    komo.addObjective([1.], ry.FS.positionDiff, ['l_gripper', 'approachframe'], ry.OT.eq, [1e1])
    komo.addObjective([2.], ry.FS.positionDiff, ['l_gripper', goal_die], ry.OT.eq, [1e1])
    komo.addObjective([3., 4.], ry.FS.positionDiff, ['l_gripper', 'approachframe'], ry.OT.eq)
    komo.addObjective([4., 5.], ry.FS.positionDiff, ['l_gripper', goal_die], ry.OT.eq, np.eye(3) - np.outer([0, 0, 1], [0, 0, 1]))
    komo.addObjective([5.], ry.FS.positionDiff, ['l_gripper', goal_die], ry.OT.eq, [1e1])
    komo.addObjective([6.], ry.FS.positionDiff, ['l_gripper', 'approachframe'], ry.OT.eq, [1e1])

    if axis == 'y':
        komo.addObjective([],  ry.FS.scalarProductXY, ['l_gripper', goal_die], ry.OT.eq, [1e1], [1.0])
        komo.addObjective([1., 2.], ry.FS.scalarProductZZ, ['l_gripper', goal_die], ry.OT.eq, [1e1], [tilt])
        komo.addObjective([4., 6.], ry.FS.scalarProductZZ, ['l_gripper', goal_die], ry.OT.eq, [1e1], [tilt])
        komo.addObjective([1., 2.], ry.FS.scalarProductXZ, [goal_die, 'l_gripper'], ry.OT.eq, [1e1], [-rot_sign * tilt])
        komo.addObjective([4., 6.], ry.FS.scalarProductXZ, [goal_die, 'l_gripper'], ry.OT.eq, [1e1], [rot_sign * tilt])

    if axis == 'x':
        if die == 1:
            komo.addObjective([], ry.FS.scalarProductXX, ['l_gripper', goal_die], ry.OT.eq, [1e1], [-1.0])
            komo.addObjective([1., 2.], ry.FS.scalarProductZZ, ['l_gripper', goal_die], ry.OT.eq, [1e1], [tilt])
            komo.addObjective([4., 6.], ry.FS.scalarProductZZ, ['l_gripper', goal_die], ry.OT.eq, [1e1], [tilt])
            komo.addObjective([1., 2.], ry.FS.scalarProductYZ, ['l_gripper', goal_die], ry.OT.eq, [1e1], [rot_sign * tilt])
            komo.addObjective([4., 6.], ry.FS.scalarProductYZ, ['l_gripper', goal_die], ry.OT.eq, [1e1], [-rot_sign * tilt])
        else:
            komo.addObjective([], ry.FS.scalarProductXX, ['l_gripper', goal_die], ry.OT.eq, [1e1], [1.0])
            komo.addObjective([1., 2.], ry.FS.scalarProductZZ, ['l_gripper', goal_die], ry.OT.eq, [1e1], [tilt])
            komo.addObjective([4., 6.], ry.FS.scalarProductZZ, ['l_gripper', goal_die], ry.OT.eq, [1e1], [tilt])
            komo.addObjective([1., 2.], ry.FS.scalarProductYZ, ['l_gripper', goal_die], ry.OT.eq, [1e1], [-rot_sign * tilt])
            komo.addObjective([4., 6.], ry.FS.scalarProductYZ, ['l_gripper', goal_die], ry.OT.eq, [1e1], [rot_sign * tilt])

    komo.addObjective([], ry.FS.distance, ['l_palm', goal_die], ry.OT.ineq, [1e1])

    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    print("KOMO report:")
    pprint(komo.report())

    if ret.feasible:
        q = komo.getPath()
        return ret, q
    else:
        print("-- Solution is infeasible!")
        return ret, None

def botop_obs_pos(bot: ry.BotOp, C: ry.Config, q_home: np.ndarray, q_obs: np.ndarray):
    bot.moveTo(q_obs[0])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

def botop_pickplace(bot: ry.BotOp, C: ry.Config, q_home: np.ndarray, q_pick: np.ndarray):

    #assert q_home.shape == (7,) and q_pick.shape == (7,) and q_place.shape == (7,), f"{q_home.shape}, {q_pick.shape}, {q_place.shape}"

    # open gripper
    bot.gripperMove(ry.ArgWord._left, .085, .1)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    # move to q_pick
    bot.moveTo(q_pick[0])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_pick[1])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    # close gripper
    #bot.gripperMove(ry.ArgWord._left, .025, .1)
    bot.gripperClose(ry.ArgWord._left, 10, 0.05, 0.1)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    bot.moveTo(q_pick[2])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_pick[3])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_pick[4])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    # open gripper
    bot.gripperMove(ry.ArgWord._left, .085, .1)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    bot.moveTo(q_pick[5])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_home)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

def check(C: ry.Config, to_del: List[Any] = [], message="") -> None:
    user_key = C.view(pause=True, message=f"{message} (Press q to quit)")
    if user_key == ord("q"):
        del C
        for o in to_del: del o
        exit(0)

def estimate_dice_pose(C, bot: ry.BotOp, q_home: np.ndarray):
    # --- 1. MOVE TO OBSERVATION POSE ---
    komo = ry.KOMO(config=C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=False)
    ret_obs, q_obs = komo_obs_pos(komo=komo, q_home=q_home)
    del komo

    if ret_obs.feasible:
        bot.moveTo(q_home)
        botop_obs_pos(bot=bot, C=C, q_home=q_home, q_obs=q_obs)
        check(C, [], "Reached Observation Pose")  # don't pass bot for deletion
    else:
        print("Observation pose infeasible")
        exit(0)

    # --- 2. VISION & POSE ESTIMATION ---
    cameras = {"top": "cameraTopMiddle", "wrist": "l_cameraWrist"}
    cam_name = cameras[args.input]
    
    if args.display and args.input is not None:
            
        num_tries = 4
        for t in range (num_tries):
            rgb, depth, pcl = bot.getImageDepthPcl(cam_name)
            # np.savez("dice_realsense_capture6.npz", rgb=rgb, depth=depth, pcl=pcl)
            # print("Saved to dice_realsense_capture.npz")

            X_world_cam_7d = C.getFrame(cam_name).getPose()
            X_world_cam = seven_d_to_matrix(X_world_cam_7d)
            fxycxy = bot.getCameraFxycxy(cameras[args.input]) 
            fx, fy, cx, cy = fxycxy
            K = np.array([[fx, 0,  cx],
                        [0,  fy, cy],
                        [0,  0,  1.0]])
            
            fig = plt.figure(figsize=(10,5))
            axs = fig.subplots(1, 2)
            axs[0].imshow(rgb)
            axs[1].matshow(depth)    
            plt.show()

            if args.real:
                quantile = 0.12
            else:
                quantile = 0.25

            solver = DicePoseEstimator(mesh_path="meshes/untitled.obj",quantile=quantile)

            poses_camera_frame = solver.estimate(pcl_xyz=pcl, pcl_rgb=rgb, visualize=True)
            if len(poses_camera_frame) == 2:
                print(f"\nSUCCESS! Found {len(poses_camera_frame)} dice.")
                break
            print(f"\nFound only {len(poses_camera_frame)} dice. \nRetrying...")
        
        if poses_camera_frame:
            T_cam_dice = poses_camera_frame
            T_world_dice = X_world_cam @ T_cam_dice

            print("Detected World Pose of Dice 1:\n", T_world_dice[0])
            print("Detected World Pose of Dice 2:\n", T_world_dice[1])
            
            pose_dice1 = matrix_to_7d(T_world_dice[0])
            pose_dice2 = matrix_to_7d(T_world_dice[1])
            C.getFrame("box1").setPose(pose_dice2) # update internal dice for decision module
            C.getFrame("box2").setPose(pose_dice1) # update internal dice for decision module
            C.getFrame("box1_axes").setPosition(pose_dice2[0:3]) # update helper axes for komo objectives
            C.getFrame("box2_axes").setPosition(pose_dice1[0:3]) # update helper axes for komo objectives
            C.view()
            print("Updated frames to detected pose.")
        else:
            print("No dice found. Using simulation ground truth.")

        bot.moveTo(q_home)
        bot.wait(C, forKeyPressed=False, forTimeToEnd=True)
        # NO del bot here



def main(args):
    ry.params_add({
        "physx/angularDamping": 0.1,
        "physx/defaultFriction": 100.0,
        "physx/defaultRestitution": 0.7,
        "botsim/verbose": 0
    })

    C = gen_scene()
    check(C, [], "Scene Loaded")
    q_home = C.getJointState().copy()
    bot = ry.BotOp(C=C, useRealRobot=args.real)

    # First observation + pose estimation
    estimate_dice_pose(C, bot, q_home)

    decision = [('y', 2), ('x', 1), ('-x', 2)]

    for rot in decision:
        # always refresh q_home from current state (small robustness tweak)
        q_home = C.getJointState().copy()

        goal_die = rot[1]
        set_approach_frame(C, goal_die)

        komo = ry.KOMO(config=C, phases=6, slicesPerPhase=1, kOrder=1, enableCollisions=True)
        ret_pick, q_pick = komo_ik(komo=komo, q_home=q_home, rot=rot)

        if ret_pick.feasible:
            check(C, [], "IK Solved. Ready to Pick?")
            botop_pickplace(bot=bot, C=C, q_home=q_home, q_pick=q_pick)
            check(C, [], "Done")
        else:
            print("Pick IK Infeasible")

        del komo

        # Update dice poses again after moving them
        estimate_dice_pose(C, bot, q_home)

    # clean shutdown
    del bot
    del C


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", default=False, help="Use real robot")
    p.add_argument("--input", type=str, default="wrist", help="cameraWrist or cameraTopMiddle")
    p.add_argument("--display", action="store_true", default=True, help="Show camera view")
    args = p.parse_args()
    main(args)
