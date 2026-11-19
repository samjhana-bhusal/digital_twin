#!/usr/bin/env python
"""S4: does simulating the plan first actually help?

    .venv/bin/python experiments/bench_loop.py [n_layouts] [tasks_per_layout]

Three arms, run on **identical layouts and identical task draws**, so every
difference between them is the thing being tested and not the luck of the seed:

  direct        one pick-and-place, executed immediately.  The S2 behaviour.
  loop          propose, roll out in the twin, repair, execute only what passed.
  loop-exact    the same loop with a twin that has the plant's true physics --
                the ablation that says how much of the loop's value depends on
                the twin being right. (Its object positions still come from the
                camera, and it still cannot see orientation; nothing here is a
                perfect twin. S5 takes those apart.)

Every arm perceives through the camera and is verified through the camera. The
reference is still resolved from ground truth; S3 is the stage that replaces
that, and keeping it fixed here means an S3 drop is attributable to grounding.

Three verdicts are recorded per episode and they are three different things:

  twin_ok   the twin predicted this plan would cleanly succeed
  task_ok   the goal predicate is true of the real final state, cleanly
  seen_ok   the verifier, looking only at a photograph, believes it succeeded
"""
from __future__ import annotations

import collections
import json
import os
import sys


import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fieldtwin.evaluate import REGIMES, paired, run
from fieldtwin.twin import TwinFidelity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARMS = {
    "direct":        None,
    "loop":          TwinFidelity(),
    "loop-exact":    TwinFidelity.exact_physics(),
}


# ---------------------------------------------------------------- reporting

def pct(a: int, b: int) -> str:
    return f"{a:4d}/{b:<4d} {100*a/max(b,1):5.1f}%"


def confusion(rows, pred_key, name, pred_true="True"):
    """A 2x2 is only informative if both margins are reported with it."""
    tp = sum(1 for r in rows if r[pred_key] and r["task_ok"])
    fp = sum(1 for r in rows if r[pred_key] and not r["task_ok"])
    fn = sum(1 for r in rows if not r[pred_key] and r["task_ok"])
    tn = sum(1 for r in rows if not r[pred_key] and not r["task_ok"])
    print(f"\n  {name} vs. what actually happened")
    print(f"                       actually clean   actually not")
    print(f"    said success  {tp:12d} {fp:15d}")
    print(f"    said failure  {fn:12d} {tn:15d}")
    n = tp + fp + fn + tn
    print(f"    agreement {100*(tp+tn)/max(n,1):.1f}%   "
          f"false confidence {fp}/{max(tp+fp,1)} of its 'success' calls")


def report(res: dict) -> None:
    rows, c = res["rows"], res["config"]
    n = len(rows)
    ok = sum(r["task_ok"] for r in rows)
    print(f"\n{'='*66}\n[{c['arm']} / {c['regime']}]  {n} episodes, "
          f"{res['elapsed_s']:.0f}s ({res['elapsed_s']/max(n,1):.2f}s each)")
    print(f"  CLEAN SUCCESS   {pct(ok, n)}")
    print(f"  goal reached    {pct(sum(r['goal_ok'] for r in rows), n)}")

    g = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        g[r["goal"]][0] += r["task_ok"]
        g[r["goal"]][1] += 1
    print("\n  by goal kind:")
    for k in sorted(g, key=lambda k: -g[k][1]):
        print(f"    {k:20s} {pct(*g[k])}")

    lab = collections.Counter(r.get("plan_label", "none") for r in rows)
    won = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        won[r.get("plan_label", "none")][0] += r["task_ok"]
        won[r.get("plan_label", "none")][1] += 1
    print("\n  plan chosen:")
    for k, _ in lab.most_common():
        print(f"    {k:20s} {pct(*won[k])}")
    sims = [r["n_sims"] for r in rows]
    if any(sims):
        print(f"    rollouts per episode: mean {np.mean(sims):.2f}, max {max(sims)}")

    # The three checks, weakest first. The skill's own report is what a system
    # without any of this machinery would have to trust.
    confusion(rows, "skill_ok", "SKILL self-report")
    if any(r["n_sims"] for r in rows):
        confusion(rows, "twin_ok", "TWIN prediction (before acting)")
    seen = [r for r in rows if r["verdict"] != "unsure"]
    for r in seen:
        r["_v"] = r["verdict"] == "success"
    if seen:
        confusion(seen, "_v", f"CAMERA verifier (after acting; judged {len(seen)}/{n})")
    print(f"  verifier declined to judge: "
          f"{sum(1 for r in rows if r['verdict']=='unsure')}/{n}")

    print("\n  failure reasons:")
    for reason, k in collections.Counter(
            r["reason"] for r in rows if not r["task_ok"]).most_common(8):
        print(f"    {k:4d}  {reason}")


def main() -> None:
    nl = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    pl = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2, 3]
    out: dict = {}
    pooled: dict = {}
    for regime in REGIMES:
        for arm in ARMS:
            rows, elapsed = [], 0.0
            for sd in seeds:
                res = run(ARMS[arm], nl, pl, seed=sd, regime=regime, label=arm)
                rows += res["rows"]
                elapsed += res["elapsed_s"]
            merged = {"rows": rows, "elapsed_s": elapsed,
                      "config": {"arm": arm, "regime": regime, "seeds": seeds,
                                 "n_layouts": nl, "per_layout": pl, **REGIMES[regime]}}
            report(merged)
            per_seed = [f"{sd}: {100*np.mean([r['task_ok'] for r in rows if r['seed']==sd]):.1f}%"
                        for sd in seeds]
            print(f"  per seed: {'  '.join(per_seed)}")
            pooled[(arm, regime)] = rows
            out[f"{arm}/{regime}"] = merged

    print(f"\n{'='*66}\nPAIRED COMPARISONS")
    for regime in REGIMES:
        print(f"\n--- {regime} ---")
        out[f"paired/loop-vs-direct/{regime}"] = paired(
            pooled[("direct", regime)], pooled[("loop", regime)], "direct", "loop")
        out[f"paired/exact-vs-loop/{regime}"] = paired(
            pooled[("loop", regime)], pooled[("loop-exact", regime)],
            "loop", "loop-exact")

    p = os.path.join(ROOT, "results", "s4_loop.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
