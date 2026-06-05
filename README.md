## Project Description
This project developed a closed-loop robotic manipulation pipeline for a dice-rotation
task. The system using Franka robotic arm integrated with Realsense camera observes
two dice, estimates their 6D poses from RGB-D point clouds, plans rotation primitives, executes robot actions, and re-perceives the scene after each step. The perception module used plane removal, DBSCAN clustering, coarse/fine ICP, and pip-based orientation refinement to recover reliable dice poses. The decision layer applied BFS over 24 discrete cube orientations to find the shortest sequence of ±90° rotations. Motion execution was handled with KOMO-based inverse kinematics, including collision, joint-limit, positioning, and orientation constraints. The project demonstrates practical experience in perception–planning–execution for sim-to-real robotic manipulation.


### Decision layer (target-sum planner)
`main.py` can plan and execute dice roll primitives (`x`, `-x`, `y`, `-y`) to reach a desired **top-face sum**.

- Run with decision layer: `python main.py --target_sum 7`
- Limit search/execution length: `python main.py --target_sum 7 --decision_max_steps 8`
- Tie-break (optional): `--decision_cost_mode steps` (fewest rolls) or `--decision_cost_mode motion` (pick the feasible first roll with lowest KOMO joint-path length)

The planner is implemented in `decision_module.py` and searches in the 24-orientation cube rotation group (per die) using BFS.

The perception is implemented in `dice_pose_estimator.py` and takes a RGB and pointcloud image as input. It returns the dice poses in camera frame.