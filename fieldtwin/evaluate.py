"""Running episodes, and comparing two ways of choosing what to do.

Extracted so that the S4 benchmark and the S5 fidelity sweep run *the same*
episode loop. If they each had their own, a difference between the two stages'
numbers could always be a difference between two harnesses, and the whole point
of S5 is to compare against S4's result.

The one knob is `fidelity`: `None` means act directly with no twin at all, and a
`TwinFidelity` means simulate, repair, then act. Everything else about an
episode is identical either way, including looking at the table before and
after.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from . import plan as planning
from . import skills, tasks, verify
from .arena import build, random_scene
from .perceive import Perception, StateView
from .robot import ARM_DOF, Panda
from .twin import Choice, Twin, TwinFidelity, choose

REGIMES = {
    "open":      {"n_cubes": 3, "min_sep": 0.085},
    "cluttered": {"n_cubes": 5, "min_sep": 0.062},
}


def run(fidelity: TwinFidelity | None, n_layouts: int = 15, per_layout: int = 5,
        seed: int = 0, regime: str = "open", budget: int = 4,
        label: str = "") -> dict:
    """One arm over one seed. Returns a row per episode."""
    cfg = REGIMES[regime]
    rows: list[dict] = []
    t0 = time.time()

    for li in range(n_layouts):
        # Seeded per layout, so arm A and arm B see the same world and the same
        # instructions. Paired comparison or no comparison.
        rng = np.random.default_rng([seed, li])
        m, sc = build(random_scene(rng, n_cubes=cfg["n_cubes"], min_sep=cfg["min_sep"]))
        d = mujoco.MjData(m)
        robot = Panda(m, d)
        robot.home()
        for _ in range(80):
            robot.step()
        rest = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())

        pool = tasks.enumerate_tasks(m, d, sc, rng)
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(per_layout, len(pool)), replace=False)

        eye = Perception(m)
        twin = Twin(sc, fidelity) if fidelity is not None else None

        for i in idx:
            d.qpos[:], d.qvel[:], d.ctrl[:] = rest
            mujoco.mj_forward(m, d)
            t = pool[int(i)]
            trng = np.random.default_rng([seed, li, int(i)])

            # --- look before acting ---------------------------------------
            before = verify.look(eye, robot, d, sc, then_ready=True)
            view = StateView(before)
            truth_start = {o.name: tasks.obj_xy(d, o.name).copy() for o in sc.objects}

            # Identity, so arms can be compared episode by episode rather than
            # rate against rate. At this sample size the absolute rate swings
            # ~7 points on the seed alone (docs/findings.md F7), so an unpaired
            # comparison of two arms is mostly measuring the draw.
            row = {"arm": label, "seed": seed, "layout": li, "task_id": int(i),
                   "regime": regime, "kind": t.kind, "ref": t.ref_kind,
                   "goal": t.goal_kind, "instruction": t.instruction,
                   "margin": t.margin, "seen_before": len(before)}

            if t.referent not in view:
                row.update(plan_label="none", n_sims=0, twin_ok=False, skill_ok=False,
                           reason="referent_not_seen", goal_ok=False, collateral_m=0.0,
                           task_ok=False, verdict="unsure", twin_said_clean=None)
                rows.append(row)
                continue

            # --- decide ---------------------------------------------------
            if twin is None:
                p = planning.direct_plan(t, m, view, sc, trng)
                choice = Choice(p, None, [], False)
            else:
                # Privileged, and only for the orientation ablation: the loop
                # proper cannot see yaw, so it is handed over only when the
                # fidelity under test says the twin is allowed to know it.
                true_yaw = ({o.name: skills.object_yaw(m, d, o.name) for o in sc.objects}
                            if fidelity.known_yaw else None)
                choice = choose(t, m, view, sc, twin, trng,
                                arm_q=robot.q_arm, ctrl=d.ctrl.copy(),
                                gripper_q=d.qpos[ARM_DOF:ARM_DOF + 2].copy(),
                                budget=budget, true_yaw=true_yaw)

            if choice.plan is None:
                row.update(plan_label="none", n_sims=choice.n_sims, twin_ok=False,
                           skill_ok=False, reason="unsatisfiable", goal_ok=False,
                           collateral_m=0.0, task_ok=False, verdict="unsure",
                           twin_said_clean=None)
                rows.append(row)
                continue

            # --- act ------------------------------------------------------
            ex = planning.execute(choice.plan, robot, m, d)
            goal_ok = bool(t.satisfied(m, d, sc))
            damage = t.collateral(m, d, sc, truth_start)
            task_ok = goal_ok and damage < tasks.COLLATERAL_TOL

            # --- check afterwards -----------------------------------------
            after = verify.look(eye, robot, d, sc)
            verdict = verify.verify(t, sc, before, after)

            row.update(plan_label=choice.plan.label, n_steps=len(choice.plan),
                       n_sims=choice.n_sims, twin_ok=bool(choice.confident),
                       # What the twin said about the plan that was actually
                       # run, which is not the same as `twin_ok`: a best-effort
                       # plan was executed despite a predicted failure.
                       twin_said_clean=(bool(choice.prediction.clean)
                                        if choice.prediction is not None else None),
                       skill_ok=bool(ex.ok), reason=ex.reason,
                       goal_ok=goal_ok, collateral_m=float(damage), task_ok=task_ok,
                       verdict=verdict.label(), seen_after=len(after))
            rows.append(row)

        eye.close()

    return {"rows": rows, "elapsed_s": time.time() - t0,
            "config": {"arm": label, "n_layouts": n_layouts, "per_layout": per_layout,
                       "seed": seed, "regime": regime, "budget": budget, **cfg}}


def run_seeds(fidelity, seeds, n_layouts, per_layout, regime, budget=4,
              label: str = "") -> list[dict]:
    rows: list[dict] = []
    for sd in seeds:
        rows += run(fidelity, n_layouts, per_layout, seed=sd, regime=regime,
                    budget=budget, label=label)["rows"]
    return rows


# ---------------------------------------------------------------- comparison

def rate(rows, key: str = "task_ok") -> float:
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def twin_agreement(rows) -> tuple[float, int, int]:
    """How often the twin's verdict on the executed plan matched reality.

    Returns (agreement, false_confidence, n). `false_confidence` counts the
    episodes the twin said would come out clean and that did not -- the
    quantity that matters, because those are the ones the system acts on.
    """
    judged = [r for r in rows if r.get("twin_said_clean") is not None]
    if not judged:
        return float("nan"), 0, 0
    agree = sum(1 for r in judged if r["twin_said_clean"] == r["task_ok"])
    wrong = sum(1 for r in judged if r["twin_said_clean"] and not r["task_ok"])
    return agree / len(judged), wrong, len(judged)


def _key(r) -> tuple:
    return (r["seed"], r["regime"], r["layout"], r["task_id"])


def paired(rows_a, rows_b, name_a: str, name_b: str, quiet: bool = False) -> dict:
    """Compare two arms episode by episode, on the episodes they both ran.

    This is the comparison that carries any claim here. Two independent rates at
    n = 300 have overlapping error bars at the effect sizes involved; the paired
    difference does not, because the layout, the instruction and the seed are
    held fixed and only the twin changes. The test is McNemar's exact, which
    looks only at the episodes where the two arms disagreed -- the ones that
    carry the information about which is better.
    """
    from scipy.stats import binomtest

    A = {_key(r): r for r in rows_a}
    B = {_key(r): r for r in rows_b}
    both = sorted(set(A) & set(B))
    a_only = sum(1 for k in both if A[k]["task_ok"] and not B[k]["task_ok"])
    b_only = sum(1 for k in both if B[k]["task_ok"] and not A[k]["task_ok"])
    agree_ok = sum(1 for k in both if A[k]["task_ok"] and B[k]["task_ok"])
    agree_no = sum(1 for k in both if not A[k]["task_ok"] and not B[k]["task_ok"])
    n_dis = a_only + b_only
    p = float(binomtest(b_only, n_dis, 0.5).pvalue) if n_dis else 1.0
    if not quiet:
        print(f"\n  PAIRED  {name_b} vs {name_a}  ({len(both)} shared episodes)")
        print(f"    both clean {agree_ok}   both failed {agree_no}")
        print(f"    {name_b} rescued {b_only}   {name_b} broke {a_only}   "
              f"net {b_only - a_only:+d}   McNemar p = {p:.2g}")
    return {"n": len(both), "rescued": b_only, "broke": a_only,
            "both_ok": agree_ok, "both_failed": agree_no, "p": p}


# ---------------------------------------------------------- fidelity, decoupled

def run_fixed_plan(fidelities: dict, n_layouts: int, per_layout: int, seed: int,
                   regime: str) -> list[dict]:
    """Ask every twin about the *same* plan, then run that plan and see.

    Why this exists: the sweep measures each twin's agreement on the plan that
    twin itself chose, and a cautious twin chooses easy-to-predict plans. That
    number is the operationally useful one -- "was the twin right about what it
    decided to do" -- but it is endogenous, and it cannot answer "is this twin a
    better model of the plant". Measured that way the *mis-specified* twin came
    out more accurate than the exact one, which is a statement about plan
    selection masquerading as one about fidelity.

    Here the plan is fixed before any twin sees it: the one-step direct plan,
    which depends only on perception. Every twin predicts that plan, the plant
    then executes it, and the predictions are scored against one outcome. Now
    accuracy is a property of the model alone, and can be put beside usefulness
    without the two contaminating each other.
    """
    cfg = REGIMES[regime]
    rows: list[dict] = []

    for li in range(n_layouts):
        rng = np.random.default_rng([seed, li])
        m, sc = build(random_scene(rng, n_cubes=cfg["n_cubes"], min_sep=cfg["min_sep"]))
        d = mujoco.MjData(m)
        robot = Panda(m, d)
        robot.home()
        for _ in range(80):
            robot.step()
        rest = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())

        pool = tasks.enumerate_tasks(m, d, sc, rng)
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(per_layout, len(pool)), replace=False)

        eye = Perception(m)
        twins = {k: Twin(sc, f) for k, f in fidelities.items()}

        for i in idx:
            d.qpos[:], d.qvel[:], d.ctrl[:] = rest
            mujoco.mj_forward(m, d)
            t = pool[int(i)]
            trng = np.random.default_rng([seed, li, int(i)])

            before = verify.look(eye, robot, d, sc, then_ready=True)
            view = StateView(before)
            if t.referent not in view:
                continue
            plan = planning.direct_plan(t, m, view, sc, trng)
            if plan is None:
                continue

            arm_q, ctrl = robot.q_arm, d.ctrl.copy()
            grip = d.qpos[ARM_DOF:ARM_DOF + 2].copy()
            said = {}
            for k, tw in twins.items():
                tw.sync(view, arm_q, ctrl, grip)
                said[k] = bool(tw.predict(plan, t).clean)

            truth_start = {o.name: tasks.obj_xy(d, o.name).copy() for o in sc.objects}
            planning.execute(plan, robot, m, d)
            goal_ok = bool(t.satisfied(m, d, sc))
            damage = t.collateral(m, d, sc, truth_start)
            rows.append({"seed": seed, "regime": regime, "layout": li,
                         "task_id": int(i), "goal": t.goal_kind,
                         "task_ok": goal_ok and damage < tasks.COLLATERAL_TOL,
                         "said": said})
        eye.close()
    return rows
