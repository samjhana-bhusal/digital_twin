"""Manipulation primitives, and how they decide whether they worked.

mink supplies inverse kinematics and MuJoCo supplies the physics. What is here
is the part no library provides: the grasp conventions, the approach/retreat
structure that keeps the arm from knocking objects over on the way in, and --
most importantly -- an explicit, checkable success condition for every skill.

A skill returns an `Outcome`, never a bare bool. The reason is the loop this
project is built around: a plan is simulated before it is executed, so the
planner needs to know *how* a skill failed (unreachable, missed the grasp, lost
the object in transit), not merely that it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .arena import CUBE_HALF, SceneSpec
from .robot import Panda, top_down

APPROACH_H = 0.11          # how far above an object to stage the approach
LIFT_H = 0.16              # carry height
GRASP_CLEARANCE = 0.001    # TCP sits this far above the object centre


@dataclass
class Outcome:
    ok: bool
    reason: str = ""
    detail: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------- world queries

def object_xy(model, data, name: str) -> np.ndarray:
    return data.body(name).xpos[:2].copy()


def object_z(model, data, name: str) -> float:
    return float(data.body(name).xpos[2])


def object_yaw(model, data, name: str) -> float:
    R = data.body(name).xmat.reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def grasp_yaw(model, data, name: str) -> float:
    """Fingers straddle a cube face. A cube has 90-degree symmetry, so fold the
    object's yaw into [-45, 45) degrees and the wrist never takes the long way
    round to reach an equivalent grasp."""
    y = object_yaw(model, data, name)
    return (y + np.pi / 4) % (np.pi / 2) - np.pi / 4


def is_grasped(robot: Panda, model, data, name: str) -> bool:
    """Held means BOTH fingers are actually touching this object.

    Earlier versions tested the finger gap plus the distance from the TCP to the
    object. That is fine when objects are far apart and wrong as soon as they
    are not: with cubes 62 mm apart, a 50 mm proximity test is satisfied by a
    *neighbour*, so the gripper could close on the wrong cube and still report
    success. Contacts are physical evidence rather than a proxy for it, and they
    cost nothing because MuJoCo has already computed them.
    """
    if robot.finger_gap() < 0.005:
        return False                       # fully closed: nothing in between
    obj = model.body(name).id
    left = model.body("left_finger").id
    right = model.body("right_finger").id
    touch_left = touch_right = False
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
        if obj not in pair:
            continue
        if left in pair:
            touch_left = True
        if right in pair:
            touch_right = True
    return touch_left and touch_right


# ---------------------------------------------------------------- primitives

def reach_above(robot: Panda, xy, yaw: float, height: float) -> Outcome:
    pos = np.array([xy[0], xy[1], height])
    res = robot.move_to(pos, top_down(yaw), steps=70)
    if not res.reached:
        return Outcome(False, "unreachable", {"pos": pos.tolist(), "ik_err": res.pos_err})
    return Outcome(True, detail={"tcp_err": res.pos_err})


def pick(robot: Panda, model, data, name: str, yaw_offset: float = 0.0) -> Outcome:
    """Approach from above, descend, close, lift. Verifies at each stage.

    `yaw_offset` rotates the wrist off the natural grasp. A cube has 90-degree
    symmetry, so an offset of pi/2 is the *same grasp on the same faces* with
    the fingers coming in along the other axis -- geometrically equivalent, and
    not at all equivalent when a neighbour happens to sit where the first pair
    of fingers wanted to go. It is the planner's one licensed way to fix a tight
    grasp without touching anything it is not allowed to touch.
    """
    xy = object_xy(model, data, name)
    yaw = grasp_yaw(model, data, name) + yaw_offset
    z0 = object_z(model, data, name)

    out = reach_above(robot, xy, yaw, z0 + APPROACH_H)
    if not out:
        return Outcome(False, "approach_unreachable", out.detail)

    robot.open_gripper()
    # descend to the object centre; re-read xy in case the approach disturbed it
    xy = object_xy(model, data, name)
    res = robot.move_to(np.array([xy[0], xy[1], z0 + GRASP_CLEARANCE]),
                        top_down(yaw), steps=50)
    if not res.reached:
        return Outcome(False, "descend_unreachable", {"ik_err": res.pos_err})

    robot.close_gripper(steps=45)
    if not is_grasped(robot, model, data, name):
        return Outcome(False, "grasp_missed",
                       {"finger_gap": robot.finger_gap(),
                        "tcp_to_obj": float(np.linalg.norm(
                            data.body(name).xpos - robot.tcp_pos()))})

    lift = robot.move_to(np.array([xy[0], xy[1], z0 + LIFT_H]), top_down(yaw), steps=50)
    if not lift.reached:
        return Outcome(False, "lift_unreachable", {})
    if object_z(model, data, name) < z0 + 0.05:
        return Outcome(False, "object_slipped", {"z": object_z(model, data, name)})
    # Height alone is not proof of a grasp: in clutter an object can ride up
    # wedged against a neighbour, or hang from a single finger. Re-check the
    # contacts so the failure is attributed here rather than surfacing later as
    # a confusing "not_holding" from place().
    if not is_grasped(robot, model, data, name):
        return Outcome(False, "lost_on_lift",
                       {"z": object_z(model, data, name),
                        "finger_gap": robot.finger_gap()})
    return Outcome(True, "picked", {"lift_z": object_z(model, data, name)})


def place(robot: Panda, model, data, name: str, target_xy,
          drop_clearance: float = 0.012) -> Outcome:
    """Carry the held object over the target and release it."""
    if not is_grasped(robot, model, data, name):
        return Outcome(False, "not_holding", {})
    tx, ty = float(target_xy[0]), float(target_xy[1])
    yaw = grasp_yaw(model, data, name)

    out = reach_above(robot, (tx, ty), yaw, CUBE_HALF + LIFT_H)
    if not out:
        return Outcome(False, "transit_unreachable", out.detail)
    if not is_grasped(robot, model, data, name):
        return Outcome(False, "dropped_in_transit", {})

    res = robot.move_to(np.array([tx, ty, CUBE_HALF + drop_clearance]),
                        top_down(yaw), steps=50)
    if not res.reached:
        return Outcome(False, "place_unreachable", {"ik_err": res.pos_err})

    robot.open_gripper(steps=30)
    robot.step(40)                                    # let it settle
    retreat = robot.move_to(np.array([tx, ty, CUBE_HALF + LIFT_H]),
                            top_down(yaw), steps=40)
    err = float(np.linalg.norm(object_xy(model, data, name) - np.array([tx, ty])))
    return Outcome(err < 0.05, "placed" if err < 0.05 else "placed_off_target",
                   {"placement_error_m": err, "retreat_ok": retreat.reached})


def pick_and_place(robot: Panda, model, data, name: str, target_xy) -> Outcome:
    got = pick(robot, model, data, name)
    if not got:
        return got
    return place(robot, model, data, name, target_xy)


def push(robot: Panda, model, data, name: str, direction, distance: float = 0.10,
         ) -> Outcome:
    """Push an object along `direction` without grasping it.

    Useful when an object is too close to a neighbour to get the fingers around,
    which is exactly the case a pick-only system fails on silently.
    """
    u = np.asarray(direction, float)[:2]
    n = np.linalg.norm(u)
    if n < 1e-9:
        return Outcome(False, "bad_direction", {})
    u = u / n
    xy = object_xy(model, data, name)
    z0 = object_z(model, data, name)
    yaw = float(np.arctan2(u[1], u[0]))

    start = xy - u * (CUBE_HALF + 0.045)
    end = xy + u * distance
    robot.close_gripper(steps=30)                     # closed fingers = a stub

    out = reach_above(robot, start, yaw, z0 + APPROACH_H)
    if not out:
        return Outcome(False, "approach_unreachable", out.detail)
    res = robot.move_to(np.array([start[0], start[1], z0]), top_down(yaw), steps=40)
    if not res.reached:
        return Outcome(False, "descend_unreachable", {})

    before = object_xy(model, data, name).copy()
    res = robot.move_to(np.array([end[0], end[1], z0]), top_down(yaw), steps=90)
    if not res.reached:
        return Outcome(False, "push_unreachable", {})
    robot.move_to(np.array([end[0], end[1], z0 + APPROACH_H]), top_down(yaw), steps=35)

    moved = object_xy(model, data, name) - before
    along = float(moved @ u)
    return Outcome(along > 0.3 * distance, "pushed",
                   {"distance_along": along, "lateral": float(moved @ np.array([-u[1], u[0]]))})
