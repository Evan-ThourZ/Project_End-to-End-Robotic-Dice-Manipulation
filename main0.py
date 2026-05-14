import argparse
import robotic as ry
import numpy as np
from typing import Tuple, List, Any
from pprint import pprint
import cv2 as cv
import math
import camera_perception1 as cp1
import camera_perception2 as cp2

def gen_scene(filename: str = "configs/pandasTable_fixedCam.g") -> ry.Config:
    """
    Set up a scene with a Panda and 2 dices.
    """

    C = ry.Config()
    #C.addFile(ry.raiPath("scenarios/pandaSingle.g"))
    C.addFile(filename)

    #base1_pos_init = [-0.18, 0.2, 0.6625]
    #base2_pos_init = [0.18, 0.2, 0.6625]
    #box1_pos_init = [-0.25, 0.22, 0.675] #sim
    #box2_pos_init = [0.25, 0.22, 0.675]  #sim
    box1_pos_init = [-0.2, 0.25, 0.675]
    box2_pos_init = [0.2, 0.25, 0.675]

    """
    f = (
        C.addFrame("base1") 
        .setPosition(base1_pos_init)
        .setShape(ry.ST.ssBox, size=[0.08, 0.08, 0.025, 0.005])
        .setColor([1.0, 1.0, 0.0])
        .setMass(1)
        .setContact(1)
    )
     f = (
        C.addFrame("base2") 
        .setPosition(base2_pos_init)
        .setShape(ry.ST.ssBox, size=[0.08, 0.08, 0.025, 0.005])
        .setColor([1.0, 1.0, 0.0])
        .setMass(1)
        .setContact(1)
    )
    """
    # Add first box
    f = (
        C.addFrame("box1") 
        .setPosition(box1_pos_init)
        .setShape(ry.ST.ssBox, size=[0.06, 0.06, 0.06, 0.005])
        .setColor([1.0, 1.0, 0.0])
        .setMass(0.1)
        .setContact(1)
        .setMeshFile("meshes/dice.obj")
    )

    # Add second box
    f = (
        C.addFrame("box2")
        .setPosition(box2_pos_init)
        .setShape(ry.ST.ssBox, size=[0.06, 0.06, 0.06, 0.005])
        .setColor([0.0, 1.0, 1.0])
        .setMass(0.1)
        .setContact(1)
        .setMeshFile("meshes/dice.obj")
    )

    # Create static frame helpers
    C.addFrame(name="box1_axes").setPosition(box1_pos_init).setShape(type=ry.ST.marker, size=[0.05])
    C.addFrame(name="box2_axes").setPosition(box2_pos_init).setShape(type=ry.ST.marker, size=[0.05])
    C.addFrame("approachframe").setPose([0., 0., 1.0, 1., 0., 0., 0.]).setShape(type=ry.ST.marker, size=[0.02])
    #C.addFrame('approachframe', f"box{args.pickbox}_axes").setRelativePosition([0.,0.,0.09]).setShape(type=ry.ST.marker, size=[0.03])

    cam = C.getFrame("l_cameraWrist")
    cam.setShape(type=ry.ST.marker, size=[0.08])
    # ADD: camera observation frame (midpoint above two dice)
    center = [
        0.5 * (box1_pos_init[0] + box2_pos_init[0]),
        0.5 * (box1_pos_init[1] + box2_pos_init[1]),
        box2_pos_init[2]+ 0.7  # 77cm in sim above top
    ]
    f = C.addFrame("camera_obs1","box1_axes").setPosition(box1_pos_init).setRelativePosition([0.,0.,0.4]).setShape(ry.ST.marker, size=[0.05]).setColor([1.0, 1.0, 1.0])
    f = C.addFrame("camera_obs2","box2_axes").setPosition(box1_pos_init).setRelativePosition([0.,0.,0.4]).setShape(ry.ST.marker, size=[0.05]).setColor([1.0, 1.0, 1.0])
    f = C.addFrame("camera_obs").setPosition(center).setShape(ry.ST.marker, size=[0.05]).setColor([1.0, 1.0, 1.0])
    
    # g_str = C.write()

    # with open(filename, 'w') as f:
    #    f.write(g_str)

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


def komo_ik(
    komo: ry.KOMO,
    q_home: np.ndarray,
    rot: tuple,
) -> Tuple[ry.SolverReturn, ry.KOMO, np.ndarray]:

    # Solve IK by defining it as a constrained optimization problem using KOMO
    # Define appropriate constraints using addObjective
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
    
    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 1, 1e-1)  # smoothness, penalizing sqr distances between consecutive configurations (velocities)
    komo.addControlObjective([], 2, 1e-1)  # acceleration
    
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

    # Solve the NLP
    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()

    # Report KOMO specs, list of objectives, constraint violations, sqr costs
    #print(f"KOMO report:")
    #pprint(komo.report())

    # Check solution feasibility
    q = None
    if ret.feasible:
        print("-- Solution is feasible")
        print(f"ret={ret}")
        q = komo.getPath()
    else:
        print("-- Solution is infeasible!")

    return ret, q


def botop_pickplace(bot: ry.BotOp, C: ry.Config, q_home: np.ndarray, q_pick: np.ndarray):

    #assert q_home.shape == (7,) and q_pick.shape == (7,) and q_place.shape == (7,), f"{q_home.shape}, {q_pick.shape}, {q_place.shape}"

    # open gripper
    bot.gripperMove(ry.ArgWord._left, width=.075, speed=.1)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    # move to q_pick
    bot.moveTo(q_pick[0])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_pick[1])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    # close gripper
    bot.gripperClose(ry.ArgWord._left, 10, 0.05, 0.1)
    #bot.gripperMove(ry.ArgWord._left, .4, .1)
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    bot.moveTo(q_pick[2])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_pick[3])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.moveTo(q_pick[4])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    # open gripper
    bot.gripperMove(ry.ArgWord._left, .075, .1)
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


def main(args):

    # Global parameters that can influence the friction, grasp force, etc.
    ry.params_add(
        {
            "physx/angularDamping": 0.1,
            "physx/defaultFriction": 100.0,
            "physx/defaultRestitution": 0.7,
            "botsim/verbose": 0,
        }
    )  # Increase verbosity to 2 for more info

    # Set up the scene
    C = gen_scene()
    check(C)
    bot = ry.BotOp(C=C, useRealRobot=args.real)
    # Store the initial joint states
    q_home = C.getJointState().copy()
    
    # add wrist camera to the scene
    pip_count = cp2.observe_and_detect_dice(C, bot, q_home, args)
    
    decision = [('y', 2), ('x', 1), ('-x', 2)]

    for rot in decision:
        # always refresh q_home from current state (small robustness tweak)
        q_home = C.getJointState().copy()

        goal_die = rot[1]
        set_approach_frame(C, goal_die)

        komo = ry.KOMO(config=C, phases=6, slicesPerPhase=1, kOrder=2, enableCollisions=False)
        ret_pick, q_pick = komo_ik(komo=komo, q_home=q_home, rot=rot)

        if ret_pick.feasible:
            check(C, [], "IK Solved. Ready to Pick?")
            botop_pickplace(bot=bot, C=C, q_home=q_home, q_pick=q_pick)
            check(C, [], "Done")
        else:
            print("Pick IK Infeasible")

        del komo

        # Update dice poses again after moving them
        pip_count = cp2.observe_and_detect_dice(C, bot, q_home, args)

    # clean shutdown
    del bot
    del C
    exit(0)


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", default=False, help="Use this arg if real robot is used")  # Use this arg to run on the real robot 
    args = p.parse_args()

    print(args)

    main(args)
