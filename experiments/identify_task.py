#!/usr/bin/env python
"""Which twin parameters can be identified, and which ones the task cares about.

    .venv/bin/python experiments/identify_task.py

[docs/findings.md](../docs/findings.md) F5 found, on the planar plant, that the
directions an identification loss constrains well are not necessarily the
directions a downstream task is sensitive to. That was an argument about a
3-DOF test rig. This script asks the same question about the system that
actually matters — the manipulator's twin — over the three parameters the S5
sweep moves: cube mass, cube sliding friction, and arm joint damping.

Two curvatures, over the same parameter space, in the same units:

  **identification** — push a cube with a scripted motion, log what a camera and
  the joint encoders would see, and finite-difference the residual with respect
  to log-parameters. The Gauss-Newton Hessian J^T J says which directions the
  data pins down. A push is used rather than a pick because a grasp is a
  discrete event: whether the fingers close on the cube flips with parameters,
  the residual jumps, and a finite difference across that jump measures the
  jump. Sliding contact is smooth, which is also why it is the classic
  system-identification manoeuvre.

  **task** — how much the loop's clean-success rate moves per unit of
  log-parameter error along each axis, measured by `sweep_fidelity.py
  directions`.

If the two orderings disagree, then "identify the twin well" and "build a twin
good enough for the job" are different objectives, and F5 generalises from the
test rig to the real system.
"""
from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fieldtwin import skills
from fieldtwin.arena import ObjectSpec, SceneSpec, build, default_scene
from fieldtwin.robot import ARM_DOF, Panda
from fieldtwin.twin import DIRECTIONS, PARAMS, Twin, TwinFidelity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Measured sensor noise, so the residual is in units of "sigmas" and the
# Hessian's eigenvalues are comparable across channels. The cube figure is the
# camera estimator's median error from S4; the joint figure is a plausible
# encoder resolution for a Panda.
SIGMA_CUBE = 0.0017        # m
SIGMA_JOINT = 1.0e-3       # rad

PUSH_DIR = (1.0, 0.35)
PUSH_DIST = 0.12


def scene() -> SceneSpec:
    """One cube, well clear of everything, so the push is the only event."""
    base = default_scene()
    return SceneSpec(objects=[ObjectSpec("red_cube", "red", (0.42, 0.02), yaw=0.0)],
                     landmarks=base.landmarks)


def push_trace(model, data, robot, sc) -> np.ndarray:
    """Run the scripted push, logging what the sensors would report."""
    log: list[np.ndarray] = []

    def tap(r):
        log.append(np.concatenate([
            data.body("red_cube").xpos[:2] / SIGMA_CUBE,
            data.qpos[:ARM_DOF] / SIGMA_JOINT,
        ]))

    robot.on_step = tap
    skills.push(robot, model, data, "red_cube", PUSH_DIR, PUSH_DIST)
    robot.on_step = None
    return np.concatenate(log)


def at(theta: np.ndarray, sc: SceneSpec, start) -> np.ndarray:
    """The trace a twin with log-parameters `theta` would produce."""
    m = np.exp(theta)
    tw = Twin(sc, TwinFidelity(float(m[0]), float(m[1]), float(m[2])))
    tw.data.qpos[:], tw.data.qvel[:], tw.data.ctrl[:] = start
    mujoco.mj_forward(tw.model, tw.data)
    return push_trace(tw.model, tw.data, tw.robot, sc)


def jacobian(sc, start, h: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """Central differences of the residual w.r.t. log-parameters."""
    base = at(np.zeros(3), sc, start)
    J = np.zeros((base.size, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        plus, minus = at(e, sc, start), at(-e, sc, start)
        n = min(len(plus), len(minus), len(base))
        J[:n, k] = (plus[:n] - minus[:n]) / (2 * h)
    return J, base


def main() -> None:
    sc = scene()
    m, _ = build(sc)
    d = mujoco.MjData(m)
    robot = Panda(m, d)
    robot.home()
    for _ in range(80):
        robot.step()
    start = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())

    truth = push_trace(m, d, robot, sc)
    print(f"reference push: {truth.size} residual channels")

    J, base = jacobian(sc, start)
    n = min(len(truth), len(base))
    print(f"residual at true parameters: RMS {np.sqrt(np.mean((truth[:n]-base[:n])**2)):.3f} sigma"
          "   (nonzero only through the twin's own build, not through physics)")

    H = J.T @ J
    w, V = np.linalg.eigh(H)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]

    print("\nIDENTIFICATION curvature (Gauss-Newton, log-parameter space)")
    print(f"  eigenvalues: {'  '.join(f'{x:.3g}' for x in w)}")
    print(f"  condition number: {w[0]/max(w[-1], 1e-300):.3g}")
    for i in range(3):
        comp = "  ".join(f"{PARAMS[k]} {V[k, i]:+.2f}" for k in range(3))
        print(f"  dir {i+1}  lambda {w[i]:10.4g}   {comp}")

    # Per-parameter identifiability: how much the residual moves for a unit of
    # log error in that parameter alone.
    print("\n  per-parameter, on its own axis:")
    sens = {}
    for k, name in enumerate(PARAMS):
        s = float(np.sqrt(H[k, k]))
        sens[name] = s
        print(f"    {name:16s} ||dr/dlog|| = {s:10.4g} sigma per unit log error")

    rank = sorted(sens, key=lambda k: -sens[k])
    print(f"\n  identification ranks them: {'  >  '.join(rank)}")

    out = {"eigenvalues": w.tolist(), "eigenvectors": V.tolist(),
           "params": list(PARAMS), "per_parameter": sens,
           "identification_rank": rank,
           "sigma": {"cube_m": SIGMA_CUBE, "joint_rad": SIGMA_JOINT}}
    p = os.path.join(ROOT, "results", "s5_identify.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {p}")
    print("\nCompare against the task ranking in results/s5_directions.json —"
          "\nsee docs/findings.md F5 for why they are not the same question.")


if __name__ == "__main__":
    main()
