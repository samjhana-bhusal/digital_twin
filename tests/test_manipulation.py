"""Invariants for the manipulation layer."""

import mujoco
import numpy as np
import pytest

from fieldtwin import skills
from fieldtwin.arena import CUBE_HALF, build, default_scene, random_scene
from fieldtwin.robot import Panda, top_down


@pytest.fixture
def world():
    m, sc = build(default_scene())
    d = mujoco.MjData(m)
    r = Panda(m, d)
    r.home()
    for _ in range(80):
        r.step()
    return m, d, r, sc


def test_scene_compiles_with_arm_and_objects(world):
    m, d, r, sc = world
    names = [m.body(i).name for i in range(m.nbody)]
    assert "hand" in names and "red_cube" in names
    assert {m.camera(i).name for i in range(m.ncam)} == {"front", "overhead", "wrist"}


def test_objects_rest_on_the_table(world):
    m, d, r, sc = world
    for ob in sc.objects:
        assert abs(skills.object_z(m, d, ob.name) - CUBE_HALF) < 3e-3


def test_ik_model_excludes_the_objects():
    """IK must not be able to 'reach' a target by sliding a cube."""
    from fieldtwin.arena import arm_model
    am = arm_model()
    assert am.nq == 9, "arm-only model should be 7 joints + 2 fingers"
    assert "red_cube" not in [am.body(i).name for i in range(am.nbody)]


def test_ik_covers_the_workspace(world):
    m, d, r, sc = world
    for x in (0.38, 0.50, 0.60):
        for y in (-0.24, 0.0, 0.24):
            _, ok, e_pos, _ = r.solve(np.array([x, y, 0.12]), top_down(0.0),
                                      q_seed=r.q_arm)
            assert ok and e_pos < 1e-3, f"({x},{y}) unreachable, err={e_pos}"


def test_top_down_points_the_gripper_at_the_table():
    for yaw in (0.0, 0.8, -1.2):
        assert np.allclose(top_down(yaw)[:, 2], [0, 0, -1], atol=1e-9)


def test_grasp_yaw_folds_into_the_cube_symmetry(world):
    m, d, r, sc = world
    y = skills.grasp_yaw(m, d, "red_cube")
    assert -np.pi / 4 - 1e-9 <= y < np.pi / 4 + 1e-9


def test_gripper_opens_and_closes(world):
    m, d, r, sc = world
    r.open_gripper()
    assert r.finger_gap() > 0.07
    r.close_gripper(60)
    assert r.finger_gap() < 0.005


def test_is_grasped_rejects_closing_on_empty_air(world):
    """Finger gap alone is not evidence: closing beside the object must not
    report a successful grasp."""
    m, d, r, sc = world
    xy = skills.object_xy(m, d, "red_cube")
    r.move_to(np.array([xy[0], xy[1] + 0.14, CUBE_HALF]), top_down(0.0), steps=60)
    r.close_gripper(45)
    assert not skills.is_grasped(r, m, d, "red_cube")


def test_pick_then_place(world):
    m, d, r, sc = world
    got = skills.pick(r, m, d, "red_cube")
    assert got.ok, got.reason
    assert skills.object_z(m, d, "red_cube") > CUBE_HALF + 0.05
    pad = sc.landmark("green_pad").pos
    put = skills.place(r, m, d, "red_cube", pad)
    assert put.ok, put.reason
    assert put.detail["placement_error_m"] < 0.05


def test_place_refuses_when_not_holding(world):
    m, d, r, sc = world
    out = skills.place(r, m, d, "red_cube", (0.5, 0.0))
    assert not out.ok and out.reason == "not_holding"


def test_push_moves_the_object_along_the_commanded_direction(world):
    m, d, r, sc = world
    before = skills.object_xy(m, d, "yellow_cube").copy()
    out = skills.push(r, m, d, "yellow_cube", direction=(1.0, 0.0), distance=0.09)
    moved = skills.object_xy(m, d, "yellow_cube") - before
    assert out.ok, out.reason
    assert moved[0] > 0.02, f"expected +x motion, got {moved}"


def test_random_scenes_keep_objects_apart():
    rng = np.random.default_rng(3)
    for _ in range(8):
        sc = random_scene(rng)
        pts = [np.array(o.pos) for o in sc.objects]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                assert np.linalg.norm(pts[i] - pts[j]) >= 0.085
