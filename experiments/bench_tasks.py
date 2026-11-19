#!/usr/bin/env python
"""S2 benchmark: can the arm carry out a well-posed instruction?

    .venv/bin/python experiments/bench_tasks.py [n_layouts] [tasks_per_layout]

The reference is resolved from ground truth here, deliberately. This measures
the *execution* half of the system in isolation, so that when a VLM grounder is
substituted in S3 the drop is attributable to grounding rather than to the arm.

Two outcomes are recorded for every episode and they are not the same thing:

  skill_ok  -- the skill believed it worked (it grasped, carried and released)
  task_ok   -- the goal predicate is actually true of the final state

Where those disagree is the interesting column. A skill that reports success on
a task that did not happen is exactly the failure mode the S4 verification loop
exists to catch.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fieldtwin import skills, tasks
from fieldtwin.arena import build, random_scene
from fieldtwin.robot import Panda

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Two regimes. "open" is the easy case the skills were built against; "cluttered"
# packs more cubes closer together than a parallel gripper comfortably fits
# between, so a plain top-down pick is no longer always available.
REGIMES = {
    "open":      {"n_cubes": 3, "min_sep": 0.085},
    "cluttered": {"n_cubes": 5, "min_sep": 0.062},
}


def run(n_layouts: int = 25, per_layout: int = 6, seed: int = 0,
        regime: str = "open") -> dict:
    rng = np.random.default_rng(seed)
    cfg = REGIMES[regime]
    rows: list[dict] = []
    t0 = time.time()

    for li in range(n_layouts):
        m, sc = build(random_scene(rng, n_cubes=cfg["n_cubes"],
                                   min_sep=cfg["min_sep"]))
        d = mujoco.MjData(m)
        robot = Panda(m, d)
        robot.home()
        for _ in range(80):
            robot.step()
        rest_qpos, rest_qvel = d.qpos.copy(), d.qvel.copy()

        pool = tasks.enumerate_tasks(m, d, sc, rng)
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(per_layout, len(pool)), replace=False)

        for i in idx:
            # every task starts from the identical settled layout
            d.qpos[:], d.qvel[:] = rest_qpos, rest_qvel
            robot.home()
            for _ in range(20):
                robot.step()
            t = pool[int(i)]
            start_xy = {o.name: tasks.obj_xy(d, o.name).copy() for o in sc.objects}

            target = t.plan_target(m, d, sc, rng)
            if target is None:
                rows.append({"kind": t.kind, "ref": t.ref_kind, "goal": t.goal_kind,
                             "skill_ok": False, "reason": "unsatisfiable",
                             "goal_ok": False, "collateral_m": 0.0,
                             "task_ok": False, "margin": t.margin})
                continue

            out = skills.pick_and_place(robot, m, d, t.referent, target)
            goal_ok = bool(t.satisfied(m, d, sc))
            damage = t.collateral(m, d, sc, start_xy)
            rows.append({
                "kind": t.kind, "ref": t.ref_kind, "goal": t.goal_kind,
                "skill_ok": bool(out.ok), "reason": out.reason,
                "goal_ok": goal_ok, "collateral_m": damage,
                # "clean" is the real bar: goal reached AND nothing else disturbed
                "task_ok": goal_ok and damage < tasks.COLLATERAL_TOL,
                "margin": t.margin, "instruction": t.instruction,
            })

    elapsed = time.time() - t0
    return {"rows": rows, "elapsed_s": elapsed,
            "config": {"n_layouts": n_layouts, "per_layout": per_layout,
                       "seed": seed, "regime": regime, **cfg}}


def report(res: dict) -> None:
    rows = res["rows"]
    n = len(rows)
    ok = sum(r["task_ok"] for r in rows)
    goal_only = sum(r["goal_ok"] for r in rows)
    messy = sum(r["goal_ok"] and not r["task_ok"] for r in rows)
    print(f"\n[{res['config']['regime']}]  {n} episodes over {res['config']['n_layouts']} layouts "
          f"in {res['elapsed_s']:.1f}s  ({res['elapsed_s']/max(n,1)*1000:.0f} ms each)")
    print(f"CLEAN SUCCESS : {ok}/{n}  ({100*ok/max(n,1):.1f}%)   "
          f"goal reached and nothing else disturbed")
    print(f"  goal reached: {goal_only}/{n}  ({100*goal_only/max(n,1):.1f}%)")
    print(f"  of those, {messy} disturbed a protected object "
          f"(>{tasks.COLLATERAL_TOL*1000:.0f} mm)")

    def table(key, title):
        print(f"\n  by {title}:")
        g = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            g[r[key]][0] += r["task_ok"]
            g[r[key]][1] += 1
        for k in sorted(g, key=lambda k: -g[k][1]):
            s, t = g[k]
            print(f"    {k:22s} {s:4d}/{t:<4d}  {100*s/t:5.1f}%")

    table("ref", "reference kind")
    table("goal", "goal kind")

    dmg = [r["collateral_m"] for r in rows if r["goal_ok"]]
    if dmg:
        print(f"  collateral on goal-reached episodes: median "
              f"{np.median(dmg)*1000:.1f} mm, p95 {np.quantile(dmg,0.95)*1000:.1f} mm")
    print("\n  failure reasons:")
    for reason, c in collections.Counter(
            r["reason"] for r in rows if not r["task_ok"]).most_common():
        print(f"    {c:4d}  {reason}")

    dis = [r for r in rows if r["skill_ok"] != r["task_ok"]]
    print(f"\n  skill/task disagreement: {len(dis)}/{n}")
    for r in collections.Counter(
            ("skill ok, task failed" if r["skill_ok"] else "skill failed, task ok")
            for r in dis).most_common():
        print(f"    {r[1]:4d}  {r[0]}")
    for r in dis[:4]:
        print(f"      e.g. [{r['reason']}] {r.get('instruction','')}")


if __name__ == "__main__":
    nl = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    pl = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    allres = {}
    for regime in REGIMES:
        res = run(nl, pl, regime=regime)
        report(res)
        allres[regime] = res
    p = os.path.join(ROOT, "results", "s2_benchmark.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(allres, f, indent=1)
    print(f"\nwrote {p}")
