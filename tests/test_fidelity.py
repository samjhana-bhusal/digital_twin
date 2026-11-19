"""Invariants for the twin's mis-specification — the S5 knobs.

Every test here exists because of one bug. `TwinFidelity` carried a `known_yaw`
flag through the whole of S4 that was never plumbed into `Twin.sync`, so the
ablation reported as "the perfect twin" differed from the ordinary one in
physics only, and the write-up claimed a confound that did not exist. A flag
that silently does nothing is worse than a missing one: it produces a result.

So the rule these tests enforce is that **every knob must be observable in the
twin's state**. Setting it has to change something you can measure, and
unsetting it has to change it back.
"""

import numpy as np
import pytest

from fieldtwin import skills
from fieldtwin.arena import ObjectSpec, build, default_scene
from fieldtwin.perceive import StateView
from fieldtwin.robot import ARM_DOF, Panda
from fieldtwin.twin import (DIRECTIONS, NOMINAL_MAGNITUDE, Twin, TwinFidelity,
                            choose)
import mujoco
from fieldtwin import tasks


def turned_scene():
    """Cubes deliberately not square to the table — the orientation knob can
    only be observed on a scene where orientation is not already zero."""
    sc = default_scene()
    for o, yaw in zip(sc.objects, (0.31, -0.24, 0.18)):
        sc.objects[sc.objects.index(o)] = ObjectSpec(o.name, o.colour, o.pos, yaw=yaw)
    return sc


@pytest.fixture(scope="module")
def world():
    m, sc = build(turned_scene())
    d = mujoco.MjData(m)
    r = Panda(m, d)
    r.home()
    for _ in range(80):
        r.step()
    return m, d, r, sc


def _yaw_in_twin(twin, name: str) -> float:
    a = twin.model.joint(f"{name}_free").qposadr[0]
    q = twin.data.qpos[a + 3:a + 7]
    return float(2 * np.arctan2(q[3], q[0]))


# ---------------------------------------------------------------- the knobs

def test_the_orientation_knob_actually_reaches_the_twin(world):
    """The regression test for the S4 bug: `known_yaw` must change the state."""
    m, d, r, sc = world
    view = StateView({o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects})
    truth = {o.name: skills.object_yaw(m, d, o.name) for o in sc.objects}
    assert abs(truth["red_cube"]) > 0.02, "fixture cube is square-on; test is vacuous"

    args = (view, r.q_arm, d.ctrl.copy(), d.qpos[ARM_DOF:ARM_DOF + 2].copy())
    blind = Twin(sc, TwinFidelity(known_yaw=False))
    blind.sync(*args, yaw=truth)
    assert abs(_yaw_in_twin(blind, "red_cube")) < 1e-9

    seeing = Twin(sc, TwinFidelity(known_yaw=True))
    seeing.sync(*args, yaw=truth)
    assert abs(_yaw_in_twin(seeing, "red_cube") - truth["red_cube"]) < 1e-6


def test_the_loop_passes_orientation_through_to_the_twin(world):
    """Plumbed at the knob is not enough; it has to survive the call chain."""
    m, d, r, sc = world
    view = StateView({o.name: d.body(o.name).xpos[:2].copy() for o in sc.objects})
    truth = {o.name: skills.object_yaw(m, d, o.name) for o in sc.objects}
    twin = Twin(sc, TwinFidelity(known_yaw=True))
    t = tasks.Task("x", tasks.ColourRef("red"), tasks.ToSideGoal("left"),
                   "red_cube", 1.0)
    choose(t, m, view, sc, twin, np.random.default_rng(0), arm_q=r.q_arm,
           ctrl=d.ctrl.copy(), gripper_q=d.qpos[ARM_DOF:ARM_DOF + 2].copy(),
           budget=1, true_yaw=truth)
    assert twin.n_rollouts >= 1
    # the twin was rewound to the synced belief, which must carry the true yaw
    assert abs(_yaw_in_twin(twin, "red_cube") - truth["red_cube"]) < 1e-6


def test_the_physics_knob_actually_reaches_the_model():
    sc = default_scene()
    plant, _ = build(sc)
    pb = plant.body("red_cube").id
    pg = plant.geom("g_red_cube").id
    for f in (TwinFidelity(), TwinFidelity.along("mass", 0.3),
              TwinFidelity.along("friction", -0.3)):
        tw = Twin(sc, f)
        tb = tw.model.body("red_cube").id
        tg = tw.model.geom("g_red_cube").id
        changed = (abs(tw.model.body_mass[tb] - plant.body_mass[pb]) > 1e-12
                   or abs(tw.model.geom_friction[tg, 0] - plant.geom_friction[pg, 0]) > 1e-12
                   or abs(tw.model.dof_damping[0] - plant.dof_damping[0]) > 1e-12)
        assert changed, f"{f.describe()} left the model untouched"


def test_exact_physics_leaves_the_model_alone():
    sc = default_scene()
    plant, _ = build(sc)
    tw = Twin(sc, TwinFidelity.exact_physics())
    assert tw.fidelity.physics_is_exact()
    b = plant.body("red_cube").id
    assert tw.model.body_mass[b] == pytest.approx(plant.body_mass[b])
    assert tw.model.geom_friction[plant.geom("g_red_cube").id, 0] == pytest.approx(
        plant.geom_friction[plant.geom("g_red_cube").id, 0])
    assert np.allclose(tw.model.dof_damping, plant.dof_damping)


# ---------------------------------------------------------------- the axis

def test_the_sweep_axis_reproduces_the_s4_default():
    """`+1x nominal` has to *be* the twin S4 ran, or the sweep has no origin."""
    a = TwinFidelity.along("nominal", NOMINAL_MAGNITUDE)
    b = TwinFidelity()
    assert a.cube_mass == pytest.approx(b.cube_mass)
    assert a.cube_friction == pytest.approx(b.cube_friction)
    assert a.joint_damping == pytest.approx(b.joint_damping)


def test_a_negative_magnitude_is_the_exact_mirror():
    """The sign test only means something if -1 undoes +1 exactly."""
    for d in DIRECTIONS:
        plus = TwinFidelity.along(d, 0.4).log_error()
        minus = TwinFidelity.along(d, -0.4).log_error()
        assert np.allclose(plus, -minus)
        assert TwinFidelity.along(d, 0.4).magnitude() == pytest.approx(
            TwinFidelity.along(d, -0.4).magnitude())


def test_magnitude_is_measured_in_log_space():
    """Equal log distance either side, so 'twice as wrong' means one thing."""
    f = TwinFidelity.along("mass", 0.5)
    assert f.magnitude() == pytest.approx(0.5)
    assert TwinFidelity.exact_physics().magnitude() == pytest.approx(0.0)
    assert TwinFidelity.along("nominal", 2 * NOMINAL_MAGNITUDE).magnitude() == \
        pytest.approx(2 * TwinFidelity().magnitude())


# ---------------------------------------------------------------- the harness

def test_the_baseline_arm_never_builds_a_twin(monkeypatch):
    """The control arm has to be a real control. If `run` could quietly build a
    twin for it, every S4 and S5 comparison would be against the wrong thing."""
    from fieldtwin import evaluate

    built = []
    real = evaluate.Twin

    class Spy(real):
        def __init__(self, *a, **k):
            built.append(1)
            super().__init__(*a, **k)

    monkeypatch.setattr(evaluate, "Twin", Spy)
    res = evaluate.run(None, n_layouts=1, per_layout=1, seed=0, regime="open")
    assert built == []
    assert res["rows"], "the harness ran no episodes, so it proved nothing"
    assert all(r["n_sims"] == 0 for r in res["rows"])
    assert all(r["twin_said_clean"] is None for r in res["rows"])


def test_the_loop_arm_records_what_the_twin_predicted():
    from fieldtwin import evaluate
    res = evaluate.run(TwinFidelity(), n_layouts=1, per_layout=1, seed=0,
                       regime="open")
    acted = [r for r in res["rows"] if r["plan_label"] != "none"]
    assert acted, "no episode got as far as acting"
    assert all(r["twin_said_clean"] is not None for r in acted)
    assert all(r["n_sims"] >= 1 for r in acted)
