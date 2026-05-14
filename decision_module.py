from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# This module plans in the *discrete* cube-rotation group (24 orientations per die).
# We keep states discrete by snapping rotations to the nearest cube rotation after each action.
#
# Convention used throughout:
#   R_world_die is a rotation matrix whose COLUMNS are the die's local axes expressed in world coords.
#   (i.e., column 0 is +X_local expressed in world, column 1 is +Y_local, column 2 is +Z_local.)
#
# Actions are +/- 90° about the die's OWN axes (body frame), so we post-multiply:
#   R' = R_world_die @ Rot_local_axis(angle)

# Face convention in the *reference* orientation:
# +Z -> 5, +Y -> 1, +X -> 3. Opposites sum to 7 (standard die).
FACE_BY_LOCAL_AXIS_SIGN: dict[tuple[str, int], int] = {
    ("x", +1): 3,
    ("x", -1): 4,
    ("y", +1): 1,
    ("y", -1): 6,
    ("z", +1): 5,
    ("z", -1): 2,
}

# Planner action: rotate exactly one die (box_id) about one of its local axes by angle_deg.
Action = Tuple[int, str, float]


def die_axis_in_world(R_world_die: np.ndarray, axis: str) -> np.ndarray:
    """
    Return the die's +axis direction expressed in world coordinates.

    (This is mainly for the convenience of debugging / visualization)
    
    """
    axis = str(axis)
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
    R = np.asarray(R_world_die, dtype=float).reshape(3, 3)
    return R[:, ("x", "y", "z").index(axis)].copy()


def top_face_from_rotation(R_world_die: np.ndarray) -> int:
    """
    Return the die's top face number given its orientation.

    "Top" means aligned with +Z_world. With the column convention, the z-component
    of each local axis is in row 2: R[2, i]. The local axis with the largest
    |z-component| is the one pointing most up/down; its sign decides +axis vs -axis.
    """
    R = np.asarray(R_world_die, dtype=float).reshape(3, 3)
    z_components = np.array([R[2, 0], R[2, 1], R[2, 2]], dtype=float)

    idx = int(np.argmax(np.abs(z_components)))
    axis = ("x", "y", "z")[idx]
    sign = 1 if float(z_components[idx]) >= 0.0 else -1
    return FACE_BY_LOCAL_AXIS_SIGN[(axis, sign)]


def sum_top_faces(rotations_world_die: Iterable[np.ndarray]) -> int:
    """Sum of top faces across multiple dice."""
    return int(sum(top_face_from_rotation(R) for R in rotations_world_die))


def _rot(axis: str, angle_deg: float) -> np.ndarray:
    """Right-handed rotation matrix about a named axis ('x'/'y'/'z')."""
    a = float(np.deg2rad(angle_deg))
    c = float(np.cos(a))
    s = float(np.sin(a))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unknown axis: {axis}")


def _R_key(R: np.ndarray) -> tuple[int, ...]:
    """
    Hashable key for a cube rotation.

    Cube rotations are integer matrices in {-1,0,1}. We round to protect against
    small numeric drift and then flatten into a tuple.
    """
    Ri = np.rint(np.asarray(R, dtype=float)).astype(int)
    return tuple(int(x) for x in Ri.reshape(-1))


@lru_cache(maxsize=1)
def all_cube_rotations() -> List[np.ndarray]:
    """
    Generate the 24 cube rotations as integer matrices.

    We do a small graph search using two generators (Rx(90), Ry(90)).
    Cached because many calls reuse the same set.
    """
    gens = [_rot("x", 90.0), _rot("y", 90.0)]
    q: deque[np.ndarray] = deque([np.eye(3)])
    seen: set[tuple[int, ...]] = set()
    rots: List[np.ndarray] = []

    while q:
        R = q.popleft()
        key = _R_key(R)
        if key in seen:
            continue
        seen.add(key)
        rots.append(np.rint(R).astype(int))

        for g in gens:
            q.append(g @ R)

    if len(rots) != 24:
        # If this trips, something is wrong with conventions or generator usage.
        raise RuntimeError(f"Expected 24 cube rotations, got {len(rots)}")
    return rots


def nearest_cube_rotation(R: np.ndarray, candidates: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """
    Snap a rotation matrix to the nearest cube rotation.

    We use trace(R^T M) as an alignment score. This keeps the planner in a small,
    discrete state space even if upstream pose estimation is noisy.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cube_rots = list(candidates) if candidates is not None else all_cube_rotations()

    best = cube_rots[0]
    best_score = float("-inf")
    for M in cube_rots:
        score = float(np.trace(R.T @ np.asarray(M, dtype=float).reshape(3, 3)))
        if score > best_score:
            best_score = score
            best = M
    return np.asarray(best, dtype=int)


def apply_die_action(
    R_world_die: np.ndarray,
    axis: str,
    angle_deg: float,
    cube_rots: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """
    Apply a +/- 90° action about the die's LOCAL axis and re-snap to the cube group.

    Why post-multiply?
      The action is in the die/body frame (local axis), so we do:
        R' = R @ Rot_local(angle)
    """
    rots = list(cube_rots) if cube_rots is not None else all_cube_rotations()
    return nearest_cube_rotation(
        np.asarray(R_world_die, dtype=float).reshape(3, 3) @ _rot(str(axis), float(angle_deg)),
        rots,
    )


def shortest_actions_to_target_sum(
    start_R_by_box: Dict[int, np.ndarray],
    target_sum: int,
    boxes: Sequence[int] = (1, 2),
    primitive_actions: Sequence[Tuple[str, float]] = (
        ("x", -90.0), ("x", 90.0),
        ("y", -90.0), ("y", 90.0),
        ("z", -90.0), ("z", 90.0),
    ),
    max_steps: Optional[int] = None,
    edge_feasible: Optional[Callable[[int, str, float], bool]] = None,
    cube_rots: Optional[Sequence[np.ndarray]] = None,
) -> Optional[List[Action]]:
    """
    BFS in the discrete cube group for multiple dice.

    Notes:
      * State is a tuple of per-die cube-rotation keys.
      * We store matrices alongside keys because the goal test depends on the "top face".
      * edge_feasible is where you inject execution constraints (collisions, gripper limits, etc.).
    """
    if not boxes:
        raise ValueError("boxes must be non-empty")

    rots = list(cube_rots) if cube_rots is not None else all_cube_rotations()

    # Snap start orientations into the discrete group so the planner is consistent.
    start_mats: List[np.ndarray] = []
    for b in boxes:
        if int(b) not in start_R_by_box:
            raise KeyError(f"Missing start rotation for box {b}")
        start_mats.append(nearest_cube_rotation(start_R_by_box[int(b)], rots))

    start_state = tuple(_R_key(R) for R in start_mats)

    q: deque[tuple[tuple[int, ...], ...]] = deque([start_state])

    # parent lets us reconstruct the shortest path after BFS finds a goal.
    parent: Dict[tuple[tuple[int, ...], ...], Optional[tuple[tuple[tuple[int, ...], ...], Action]]] = {
        start_state: None
    }
    mats: Dict[tuple[tuple[int, ...], ...], List[np.ndarray]] = {start_state: start_mats}

    def depth_of(state: tuple[tuple[int, ...], ...]) -> int:
        # Kept simple; typical plans are short. If you scale this up, store depth explicitly.
        d = 0
        p = parent[state]
        while p is not None:
            d += 1
            p = parent[p[0]]
        return d

    while q:
        cur_state = q.popleft()
        cur_mats = mats[cur_state]

        if sum_top_faces(cur_mats) == int(target_sum):
            path: List[Action] = []
            s = cur_state
            while parent[s] is not None:
                prev, action = parent[s]
                path.append(action)
                s = prev
            path.reverse()
            return path

        if max_steps is not None and depth_of(cur_state) >= int(max_steps):
            continue

        for i, box_id in enumerate(boxes):
            for axis, angle in primitive_actions:
                axis = str(axis)
                angle = float(angle)

                if edge_feasible is not None and (not edge_feasible(int(box_id), axis, angle)):
                    continue

                nxt_mats = list(cur_mats)
                nxt_mats[i] = apply_die_action(nxt_mats[i], axis, angle, rots)
                nxt_state = tuple(_R_key(R) for R in nxt_mats)

                if nxt_state in parent:
                    continue

                parent[nxt_state] = (cur_state, (int(box_id), axis, angle))
                mats[nxt_state] = nxt_mats
                q.append(nxt_state)

    return None


def candidate_first_actions_to_target_sum(
    start_R_by_box: Dict[int, np.ndarray],
    target_sum: int,
    boxes: Sequence[int] = (1, 2),
    primitive_actions: Sequence[Tuple[str, float]] = (
        ("x", -90.0), ("x", 90.0),
        ("y", -90.0), ("y", 90.0),
        ("z", -90.0), ("z", 90.0),
    ),
    max_steps: Optional[int] = None,
    edge_feasible: Optional[Callable[[int, str, float], bool]] = None,
    cube_rots: Optional[Sequence[np.ndarray]] = None,
) -> List[Action]:
    """
    Return all distinct FIRST actions that participate in some optimal (shortest) solution.

    Useful when multiple equally-short plans exist and you want to defer tie-breaking
    to another layer (e.g., motion cost, uncertainty, safety margins).
    """
    if not boxes:
        raise ValueError("boxes must be non-empty")

    rots = list(cube_rots) if cube_rots is not None else all_cube_rotations()

    start_mats: List[np.ndarray] = []
    for b in boxes:
        if int(b) not in start_R_by_box:
            raise KeyError(f"Missing start rotation for box {b}")
        start_mats.append(nearest_cube_rotation(start_R_by_box[int(b)], rots))

    start_state = tuple(_R_key(R) for R in start_mats)
    if sum_top_faces(start_mats) == int(target_sum):
        return []

    q: deque[tuple[tuple[int, ...], ...]] = deque([start_state])

    parent: Dict[tuple[tuple[int, ...], ...], Optional[tuple[tuple[tuple[int, ...], ...], Action]]] = {
        start_state: None
    }
    mats: Dict[tuple[tuple[int, ...], ...], List[np.ndarray]] = {start_state: start_mats}
    depth: Dict[tuple[tuple[int, ...], ...], int] = {start_state: 0}

    best_solution_depth: Optional[int] = None
    solution_states: List[tuple[tuple[int, ...], ...]] = []

    while q:
        cur_state = q.popleft()
        cur_depth = depth[cur_state]
        cur_mats = mats[cur_state]

        if best_solution_depth is not None and cur_depth > best_solution_depth:
            break

        if sum_top_faces(cur_mats) == int(target_sum):
            best_solution_depth = cur_depth
            solution_states.append(cur_state)
            continue

        if max_steps is not None and cur_depth >= int(max_steps):
            continue

        for i, box_id in enumerate(boxes):
            for axis, angle in primitive_actions:
                axis = str(axis)
                angle = float(angle)

                if edge_feasible is not None and (not edge_feasible(int(box_id), axis, angle)):
                    continue

                nxt_mats = list(cur_mats)
                nxt_mats[i] = apply_die_action(nxt_mats[i], axis, angle, rots)
                nxt_state = tuple(_R_key(R) for R in nxt_mats)

                if nxt_state in parent:
                    continue

                parent[nxt_state] = (cur_state, (int(box_id), axis, angle))
                mats[nxt_state] = nxt_mats
                depth[nxt_state] = cur_depth + 1
                q.append(nxt_state)

    if best_solution_depth is None:
        return []

    first_actions: dict[Action, None] = {}
    for s in solution_states:
        cur = s
        first: Optional[Action] = None

        # Walk back to the start; the last "action" we see is the first step from start.
        while parent[cur] is not None:
            prev, action = parent[cur]
            first = action
            cur = prev

        if first is not None:
            first_actions[first] = None

    return list(first_actions.keys())


def actions_to_rai_main_primitives(actions: Sequence[Action]) -> List[Tuple[str, int]]:
    """
    Convert planner actions into the primitive format expected by the RAI main layer.

    Output:
      (axis_raw, die_id) where axis_raw in {"x","-x","y","-y","z","-z"}.
    """
    out: List[Tuple[str, int]] = []
    for box_id, axis, angle in actions:
        axis = str(axis)
        angle = float(angle)

        if axis not in ("x", "y", "z"):
            raise ValueError(f"Only x/y/z primitives are supported, got axis={axis}")
        if abs(abs(angle) - 90.0) > 1e-6:
            raise ValueError(f"Primitive expects +/-90deg, got angle={angle}")

        axis_raw = f"-{axis}" if angle < 0.0 else axis
        out.append((axis_raw, int(box_id)))
    return out


def plan_rai_main_primitives(
    start_R_by_box: Dict[int, np.ndarray],
    target_sum: int,
    boxes: Sequence[int] = (1, 2),
    primitive_actions: Sequence[Tuple[str, float]] = (
        ("x", -90.0), ("x", 90.0),
        ("y", -90.0), ("y", 90.0),
        ("z", -90.0), ("z", 90.0),
    ),
    max_steps: Optional[int] = 8,
) -> Optional[List[Tuple[str, int]]]:
    """Thin wrapper: plan in (box, axis, angle) space and convert to RAI primitives."""
    actions = shortest_actions_to_target_sum(
        start_R_by_box=start_R_by_box,
        target_sum=int(target_sum),
        boxes=boxes,
        primitive_actions=primitive_actions,
        max_steps=max_steps,
    )
    if actions is None:
        return None
    return actions_to_rai_main_primitives(actions)