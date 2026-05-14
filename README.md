## rai_main_merge

### Decision layer (target-sum planner)
`main.py` can plan and execute dice roll primitives (`x`, `-x`, `y`, `-y`) to reach a desired **top-face sum**.

- Run with decision layer: `python main.py --target_sum 7`
- Limit search/execution length: `python main.py --target_sum 7 --decision_max_steps 8`
- Tie-break (optional): `--decision_cost_mode steps` (fewest rolls) or `--decision_cost_mode motion` (pick the feasible first roll with lowest KOMO joint-path length)

The planner is implemented in `decision_module.py` and searches in the 24-orientation cube rotation group (per die) using BFS.

The perception is implemented in `dice_pose_estimator.py` and takes a RGB and pointcloud image as input. It returns the dice poses in camera frame.