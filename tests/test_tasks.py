"""Invariants for the task layer.

These matter more than they look. The benchmark's number is only meaningful if
the instructions it scores are well-posed, so the generator's rejection rules
get tested as carefully as the robot does.
"""

import mujoco
import numpy as np
import pytest

from fieldtwin import tasks
from fieldtwin.arena import ObjectSpec, SceneSpec, build, default_scene, random_scene
from fieldtwin.robot import Panda


def settled(scene):
    m, sc = build(scene)
    d = mujoco.MjData(m)
    r = Panda(m, d)
    r.home()
    for _ in range(80):
        r.step()
    return m, d, r, sc


@pytest.fixture
def world():
    return settled(default_scene())


def line_scene(positions):
    """A scene with cubes at exactly the given (x, y), for convention tests."""
    base = default_scene()
    objs = [ObjectSpec(f"{c}_cube", c, p)
            for c, p in zip(["red", "blue", "yellow"], positions)]
    return SceneSpec(objects=objs, landmarks=base.landmarks)


# ---------------------------------------------------------------- conventions

def test_left_is_plus_y_in_the_robot_frame():
    """The robot faces +x, so its left is +y. Everything spatial depends on
    this, so it is pinned rather than assumed."""
    m, d, r, sc = settled(line_scene([(0.46, 0.18), (0.46, 0.00), (0.46, -0.18)]))
    ref = tasks.SideOfObjectRef("blue", "left")
    name, margin = ref.resolve(m, d, sc)
    assert name == "red_cube" and margin > 0
    ref2 = tasks.SideOfObjectRef("blue", "right")
    assert ref2.resolve(m, d, sc)[0] == "yellow_cube"


def test_superlatives_match_the_convention():
    m, d, r, sc = settled(line_scene([(0.46, 0.18), (0.46, 0.00), (0.46, -0.18)]))
    assert tasks.SuperlativeRef("leftmost").resolve(m, d, sc)[0] == "red_cube"
    assert tasks.SuperlativeRef("rightmost").resolve(m, d, sc)[0] == "yellow_cube"


def test_closest_and_farthest_are_measured_from_the_base():
    m, d, r, sc = settled(line_scene([(0.38, 0.0), (0.50, 0.12), (0.60, -0.10)]))
    assert tasks.SuperlativeRef("closest").resolve(m, d, sc)[0] == "red_cube"
    assert tasks.SuperlativeRef("farthest").resolve(m, d, sc)[0] == "yellow_cube"


# ---------------------------------------------------------------- rejection rules

def test_ambiguous_side_reference_is_rejected():
    """Two cubes on the same side of the anchor: the phrase means nothing."""
    m, d, r, sc = settled(line_scene([(0.44, 0.16), (0.56, 0.14), (0.46, -0.16)]))
    name, _ = tasks.SideOfObjectRef("yellow", "left").resolve(m, d, sc)
    assert name is None, "two cubes left of the anchor must not resolve"


def test_near_tie_is_rejected_by_margin():
    """A reference that wins by a hair is a coin flip, not an instruction."""
    m, d, r, sc = settled(line_scene([(0.46, 0.10), (0.46, 0.09), (0.46, -0.20)]))
    rng = np.random.default_rng(0)
    for t in tasks.enumerate_tasks(m, d, sc, rng):
        assert t.margin >= tasks.MIN_MARGIN


def test_already_satisfied_goals_are_not_generated(world):
    m, d, r, sc = world
    rng = np.random.default_rng(0)
    for t in tasks.enumerate_tasks(m, d, sc, rng):
        assert not t.satisfied(m, d, sc), f"{t.instruction} is already true"


def test_no_self_referential_next_to(world):
    m, d, r, sc = world
    rng = np.random.default_rng(0)
    for t in tasks.enumerate_tasks(m, d, sc, rng):
        if isinstance(t.goal, tasks.NextToGoal):
            assert t.goal._other() != t.referent


def test_colour_reference_is_exact(world):
    m, d, r, sc = world
    for colour in ("red", "blue", "yellow"):
        name, margin = tasks.ColourRef(colour).resolve(m, d, sc)
        assert name == f"{colour}_cube" and margin == np.inf


# ---------------------------------------------------------------- planning

def test_plan_target_satisfies_the_goal_it_came_from(world):
    """A planned placement must actually make the predicate true if reached."""
    m, d, r, sc = world
    rng = np.random.default_rng(1)
    checked = 0
    for t in tasks.enumerate_tasks(m, d, sc, rng):
        q = t.plan_target(m, d, sc, rng)
        assert q is not None
        # teleport the referent there and re-evaluate the predicate
        jid = m.body(t.referent).jntadr[0]
        adr = m.jnt_qposadr[jid]
        saved = d.qpos[adr:adr + 3].copy()
        d.qpos[adr:adr + 2] = q
        mujoco.mj_forward(m, d)
        ok = t.satisfied(m, d, sc)
        d.qpos[adr:adr + 3] = saved
        mujoco.mj_forward(m, d)
        assert ok, f"planned target does not satisfy: {t.instruction}"
        checked += 1
    assert checked > 20


def test_plan_target_keeps_clear_of_other_objects(world):
    """...except for occupied-spot tasks, whose whole point is a blocked
    destination. Those are checked separately."""
    m, d, r, sc = world
    rng = np.random.default_rng(2)
    checked = 0
    for t in tasks.enumerate_tasks(m, d, sc, rng):
        if getattr(t.goal, "requires_clearing", False):
            continue
        q = t.plan_target(m, d, sc, rng)
        for o in sc.objects:
            if o.name == t.referent:
                continue
            assert np.linalg.norm(q - tasks.obj_xy(d, o.name)) >= tasks.CLEARANCE
        checked += 1
    assert checked > 15


def test_every_reference_kind_appears_across_layouts():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(25):
        m, d, r, sc = settled(random_scene(rng))
        for t in tasks.enumerate_tasks(m, d, sc, rng):
            seen.add(t.ref_kind)
    assert seen == set(tasks.REFERENCE_KIND.values()), f"missing: {set(tasks.REFERENCE_KIND.values())-seen}"


# ---------------------------------------------------------------- occupied-spot tasks

def test_occupied_spot_goal_captures_the_spot_not_the_object(world):
    """The destination must be a fixed point in space. If it tracked the
    blocker, moving the blocker out of the way would move the goal with it and
    the task could never be completed."""
    m, d, r, sc = world
    spot = tuple(tasks.obj_xy(d, "blue_cube"))
    goal = tasks.AtObjectSpotGoal("blue", spot)
    # shove the blocker aside
    jid = m.body("blue_cube").jntadr[0]
    adr = m.jnt_qposadr[jid]
    d.qpos[adr:adr + 2] = np.array(spot) + np.array([0.0, 0.20])
    mujoco.mj_forward(m, d)
    assert goal.spot == spot
    # the red cube placed at the captured spot still satisfies the goal
    jid2 = m.body("red_cube").jntadr[0]
    adr2 = m.jnt_qposadr[jid2]
    d.qpos[adr2:adr2 + 2] = np.array(spot)
    mujoco.mj_forward(m, d)
    assert goal.satisfied(m, d, sc, "red_cube")


def test_occupied_spot_tasks_are_generated(world):
    m, d, r, sc = world
    rng = np.random.default_rng(0)
    kinds = {t.goal_kind for t in tasks.enumerate_tasks(m, d, sc, rng)}
    assert "at-occupied-spot" in kinds


def test_protected_exempts_the_blocker_but_nothing_else(world):
    m, d, r, sc = world
    spot = tuple(tasks.obj_xy(d, "blue_cube"))
    t = tasks.Task("x", tasks.ColourRef("red"),
                   tasks.AtObjectSpotGoal("blue", spot), "red_cube", np.inf)
    prot = set(t.protected(sc))
    assert "blue_cube" not in prot, "the blocker has to move"
    assert "red_cube" not in prot, "the referent is what we are moving"
    assert "yellow_cube" in prot


def test_protected_covers_everything_for_ordinary_goals(world):
    m, d, r, sc = world
    t = tasks.Task("x", tasks.ColourRef("red"), tasks.OnPadGoal("green_pad"),
                   "red_cube", np.inf)
    assert set(t.protected(sc)) == {"blue_cube", "yellow_cube"}


def test_collateral_measures_the_worst_displacement(world):
    m, d, r, sc = world
    t = tasks.Task("x", tasks.ColourRef("red"), tasks.OnPadGoal("green_pad"),
                   "red_cube", np.inf)
    start = {o.name: tasks.obj_xy(d, o.name).copy() for o in sc.objects}
    assert t.collateral(m, d, sc, start) < 1e-6
    jid = m.body("yellow_cube").jntadr[0]
    adr = m.jnt_qposadr[jid]
    d.qpos[adr] += 0.07
    mujoco.mj_forward(m, d)
    assert abs(t.collateral(m, d, sc, start) - 0.07) < 1e-3


def test_occupied_spot_target_is_allowed_to_be_blocked(world):
    """plan_target normally refuses occupied destinations. These tasks are the
    exception, or they could never be generated at all."""
    m, d, r, sc = world
    rng = np.random.default_rng(0)
    spot = tuple(tasks.obj_xy(d, "blue_cube"))
    t = tasks.Task("x", tasks.ColourRef("red"),
                   tasks.AtObjectSpotGoal("blue", spot), "red_cube", np.inf)
    q = t.plan_target(m, d, sc, rng)
    assert q is not None and np.allclose(q, spot)
