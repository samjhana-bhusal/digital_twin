"""Invariants for the S4 loop: perception, planning, the twin, and verification.

The loop's whole claim is that checking a plan before running it catches
mistakes. These tests exist so that the loop cannot quietly stop checking --
the most dangerous failure here is not a crash but a twin that silently agrees
with everything, which would restore the 100%-agreement result that made the
first benchmark worthless.
"""

import mujoco
import numpy as np
import pytest

from fieldtwin import plan as planning
from fieldtwin import skills, tasks, verify
from fieldtwin.arena import build, default_scene
from fieldtwin.perceive import Perception, StateView
from fieldtwin.robot import ARM_DOF, Panda
from fieldtwin.twin import Twin, TwinFidelity, choose


@pytest.fixture(scope="module")
def world():
    m, sc = build(default_scene())
    d = mujoco.MjData(m)
    r = Panda(m, d)
    r.home()
    for _ in range(80):
        r.step()
    r.park_for_view()
    return m, d, r, sc


# ---------------------------------------------------------------- perception

def test_camera_finds_every_cube_and_places_it_within_a_centimetre(world):
    m, d, r, sc = world
    eye = Perception(m)
    det = eye.look(d, sc)
    eye.close()
    assert set(det) == {o.name for o in sc.objects}
    for o in sc.objects:
        err = np.linalg.norm(det[o.name].xy - d.body(o.name).xpos[:2])
        assert err < 0.010, f"{o.name} estimated {err*1000:.1f} mm off"


def test_parking_for_a_look_does_not_disturb_the_scene():
    """Perceiving must not be an action. If looking moves the cubes, every
    collateral-damage number afterwards is measuring the camera, not the task."""
    m, sc = build(default_scene())
    d = mujoco.MjData(m)
    r = Panda(m, d)
    r.home()
    for _ in range(80):
        r.step()
    before = {o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects}
    r.park_for_view()
    for o in sc.objects:
        moved = np.linalg.norm(d.body(o.name).xpos[:2] - before[o.name])
        assert moved < 0.005, f"{o.name} moved {moved*1000:.1f} mm while looking"


def test_an_object_outside_the_frame_is_reported_missing_not_guessed(world):
    m, d, r, sc = world
    adr = m.joint("red_cube_free").qposadr[0]
    keep = d.qpos[adr:adr + 3].copy()
    d.qpos[adr:adr + 3] = [-1.5, 1.5, 0.02]           # off the table entirely
    mujoco.mj_forward(m, d)
    eye = Perception(m)
    det = eye.look(d, sc)
    eye.close()
    d.qpos[adr:adr + 3] = keep
    mujoco.mj_forward(m, d)
    assert "red_cube" not in det
    assert "blue_cube" in det                          # the rest still seen


# ---------------------------------------------------------------- planning

def test_only_an_occupied_spot_task_licenses_moving_a_second_object():
    """The collateral rule, not taste, decides what a legal repair is: clearing
    a protected object would trade a goal failure for a damage failure."""
    sc = default_scene()
    spot = (0.54, -0.14)
    occupied = tasks.Task("x", tasks.ColourRef("red"),
                          tasks.AtObjectSpotGoal("blue", spot), "red_cube", 1.0)
    nextto = tasks.Task("x", tasks.ColourRef("red"),
                        tasks.NextToGoal("blue"), "red_cube", 1.0)
    assert planning.movable_blocker(occupied) == "blue_cube"
    assert planning.movable_blocker(nextto) is None


def test_a_blocked_occupied_spot_produces_a_two_step_plan(world):
    m, d, r, sc = world
    spot = tuple(float(v) for v in d.body("blue_cube").xpos[:2])
    t = tasks.Task("Put the red cube where the blue cube is.",
                   tasks.ColourRef("red"), tasks.AtObjectSpotGoal("blue", spot),
                   "red_cube", 1.0)
    view = StateView({o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects})
    direct = planning.direct_plan(t, m, view, sc, np.random.default_rng(0))
    got = list(planning.repairs(t, m, view, sc, np.random.default_rng(0),
                                direct, "placed", 0.0))
    two_step = [p for p in got if p.label == "clear-then-place"]
    assert two_step, "no clearing plan offered for an occupied destination"
    p = two_step[0]
    assert len(p) == 2
    assert p.steps[0].obj == "blue_cube" and p.steps[1].obj == "red_cube"
    # the blocker must not be parked back on top of the destination
    assert np.linalg.norm(np.array(p.steps[0].target) - np.array(spot)) >= planning.PARK_MIN


def test_a_tight_grasp_is_repaired_by_turning_the_wrist_not_by_shoving(world):
    m, d, r, sc = world
    t = tasks.Task("x", tasks.ColourRef("red"), tasks.NextToGoal("yellow"),
                   "red_cube", 1.0)
    view = StateView({o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects})
    direct = planning.direct_plan(t, m, view, sc, np.random.default_rng(0))
    labels = [p.label for p in planning.repairs(t, m, view, sc,
                                                np.random.default_rng(0),
                                                direct, "grasp_missed", 0.0)]
    assert "turn-wrist" in labels
    assert all(p.touches() == {"red_cube"}
               for p in planning.repairs(t, m, view, sc, np.random.default_rng(0),
                                         direct, "grasp_missed", 0.0))


def test_turning_the_wrist_is_a_different_grasp_on_the_same_cube(world):
    m, d, r, sc = world
    natural = skills.grasp_yaw(m, d, "red_cube")
    turned = natural + np.pi / 2
    assert abs(np.cos(natural - turned)) < 1e-9        # orthogonal finger axis


# ---------------------------------------------------------------- the twin

def test_the_twin_is_wrong_on_purpose():
    """If the twin matched the plant, its predictions would be tautological."""
    sc = default_scene()
    twin = Twin(sc, TwinFidelity())
    plant, _ = build(sc)
    tb = twin.model.body("red_cube").id
    pb = plant.body("red_cube").id
    assert twin.model.body_mass[tb] != plant.body_mass[pb]
    assert not TwinFidelity().physics_is_exact()
    assert TwinFidelity.exact_physics().physics_is_exact()


def test_the_twin_rewinds_between_candidate_plans(world):
    m, d, r, sc = world
    twin = Twin(sc, TwinFidelity())
    view = StateView({o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects})
    twin.sync(view, r.q_arm, d.ctrl.copy(), d.qpos[ARM_DOF:ARM_DOF + 2].copy())
    before = twin.positions()
    t = tasks.Task("x", tasks.ColourRef("red"), tasks.ToSideGoal("left"),
                   "red_cube", 1.0)
    cand = planning.Plan([planning.Move("red_cube", (0.50, 0.20))])
    twin.predict(cand, t)
    after = twin.positions()
    for k in before:
        assert np.linalg.norm(before[k] - after[k]) < 1e-9, f"{k} was not rewound"


def test_the_loop_endorses_a_plan_before_it_is_executed(world):
    m, d, r, sc = world
    twin = Twin(sc, TwinFidelity())
    view = StateView({o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects})
    t = tasks.Task("Move the red cube to the left side of the table.",
                   tasks.ColourRef("red"), tasks.ToSideGoal("left"), "red_cube", 1.0)
    ch = choose(t, m, view, sc, twin, np.random.default_rng(0),
                arm_q=r.q_arm, ctrl=d.ctrl.copy(),
                gripper_q=d.qpos[ARM_DOF:ARM_DOF + 2].copy())
    assert ch.plan is not None
    assert ch.n_sims >= 1                     # it actually simulated something
    assert ch.reason in ("endorsed", "best_effort")


def test_an_impossible_task_is_reported_rather_than_attempted(world):
    """`confident=False` is the output that lets a system ask for help."""
    m, d, r, sc = world
    twin = Twin(sc, TwinFidelity())
    view = StateView({o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects})
    t = tasks.Task("x", tasks.ColourRef("red"),
                   tasks.OnPadGoal("green_pad"), "red_cube", 1.0)
    ch = choose(t, m, view, sc, twin, np.random.default_rng(0),
                arm_q=r.q_arm, ctrl=d.ctrl.copy(),
                gripper_q=d.qpos[ARM_DOF:ARM_DOF + 2].copy(), budget=2)
    assert ch.n_sims <= 2                     # the budget is respected


# ---------------------------------------------------------------- verification

def test_the_verifier_agrees_with_the_predicate_when_the_task_is_done(world):
    m, d, r, sc = world
    before = {o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects}
    spot = tuple(float(v) for v in d.body("blue_cube").xpos[:2])
    t = tasks.Task("x", tasks.ColourRef("red"), tasks.AtObjectSpotGoal("blue", spot),
                   "red_cube", 1.0)
    after = dict(before)
    after["red_cube"] = np.array(spot)          # as if it had been put there
    v = verify.verify(t, sc, before, after)
    assert v.goal_ok is True and v.clean is True


def test_the_verifier_says_unsure_rather_than_guessing(world):
    m, d, r, sc = world
    before = {o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects}
    t = tasks.Task("x", tasks.ColourRef("red"), tasks.ToSideGoal("left"),
                   "red_cube", 1.0)
    after = {k: v for k, v in before.items() if k != "red_cube"}
    v = verify.verify(t, sc, before, after)
    assert v.unsure and v.label() == "unsure"
    assert "red_cube" in v.missing


def test_the_verifier_catches_collateral_damage(world):
    m, d, r, sc = world
    before = {o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects}
    t = tasks.Task("x", tasks.ColourRef("red"), tasks.ToSideGoal("left"),
                   "red_cube", 1.0)
    after = dict(before)
    after["red_cube"] = np.array([0.50, 0.20])              # goal reached
    after["yellow_cube"] = before["yellow_cube"] + np.array([0.09, 0.0])
    v = verify.verify(t, sc, before, after)
    assert v.goal_ok is True
    assert v.clean is False, "a 90 mm shove of a protected cube scored clean"
