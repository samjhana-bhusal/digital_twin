#!/usr/bin/env python
"""S5: how wrong is the twin allowed to be — and is wrongness even the axis?

    .venv/bin/python experiments/sweep_fidelity.py factorial [nl] [pl] [seeds]
    .venv/bin/python experiments/sweep_fidelity.py sweep     [nl] [pl] [seeds]
    .venv/bin/python experiments/sweep_fidelity.py directions[nl] [pl] [seeds]

S4 asked for an easy ablation and got a surprise: a twin given the plant's true
physics chose *worse* plans than the mis-specified one. This stage takes that
apart.

**factorial** — the twin's two kinds of ignorance, crossed. Physics
(mis-specified or exact) against orientation (unseen or known). S4 reported
these as one knob; they were not even both connected, so this is the first
honest measurement of either.

**sweep** — the sign test, and the heart of the stage. The mis-specification is
a direction in log-parameter space with a signed magnitude, so -1 is the exact
mirror of +1: where the default twin believes cubes are heavier and more
slippery than they are, its mirror believes they are lighter and grippier. Two
hypotheses make opposite predictions here, and one run separates them:

    if what helps is the twin's PESSIMISM  -> the curve rises to the right
    if what helps is merely that it DISAGREES -> the curve dips at zero and
                                                 rises on both sides

**directions** — which parameter is actually doing the work. The manipulation
analogue of the stiff/flat question in docs/findings.md F5: an error only
matters in the directions the *task* is sensitive to, and that is not the same
set as the directions identification data constrains.

Every arm runs identical layouts, instructions and seeds, so every comparison is
paired (F7 — the absolute rate swings ~7 points on the seed alone).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fieldtwin.evaluate import (paired, rate, run_fixed_plan, run_seeds,
                                twin_agreement)
from fieldtwin.twin import NOMINAL_MAGNITUDE, TwinFidelity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIME = "cluttered"          # where S4's effect was significant; see F8

# In units of the S4 default's mis-specification, which is 0.22 in log space.
SWEEP = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)


def summarise(label: str, rows: list[dict]) -> dict:
    agree, wrong, n_judged = twin_agreement(rows)
    two_step = np.mean([r.get("plan_label") == "clear-then-place" for r in rows])
    occ = [r for r in rows if r["goal"] == "at-occupied-spot"]
    out = {
        "label": label, "n": len(rows),
        "clean": rate(rows),
        "occupied_spot": rate(occ) if occ else float("nan"),
        "twin_agreement": agree,
        "twin_false_confidence": wrong / max(n_judged, 1),
        "two_step_share": float(two_step),
        "sims": float(np.mean([r.get("n_sims", 0) for r in rows])),
    }
    print(f"  {label:34s} clean {100*out['clean']:5.1f}%  "
          f"occupied {100*out['occupied_spot']:5.1f}%  "
          f"twin-agree {100*out['twin_agreement']:5.1f}%  "
          f"false-conf {100*out['twin_false_confidence']:4.1f}%  "
          f"2-step {100*out['two_step_share']:4.1f}%")
    return out


def go(fidelity, label, nl, pl, seeds) -> list[dict]:
    return run_seeds(fidelity, seeds, nl, pl, REGIME, label=label)


# ---------------------------------------------------------------- experiments

def factorial(nl, pl, seeds) -> dict:
    print(f"\n=== FACTORIAL ({REGIME}) — physics x orientation ===")
    cells = {
        "direct (no twin)":            None,
        "exact physics, yaw unseen":   TwinFidelity.exact_physics(known_yaw=False),
        "exact physics, yaw known":    TwinFidelity.exact_physics(known_yaw=True),
        "mis-specified, yaw unseen":   TwinFidelity(),
        "mis-specified, yaw known":    TwinFidelity(known_yaw=True),
    }
    rows = {k: go(v, k, nl, pl, seeds) for k, v in cells.items()}
    stats = {k: summarise(k, v) for k, v in rows.items()}

    print("\n  paired comparisons:")
    comps = {}
    base = "exact physics, yaw unseen"
    for k in cells:
        if k == base:
            continue
        comps[k] = paired(rows[base], rows[k], base, k)
    return {"stats": stats, "paired": comps,
            "rows": {k: v for k, v in rows.items()}}


def sweep(nl, pl, seeds) -> dict:
    print(f"\n=== SWEEP ({REGIME}) — signed mis-specification along `nominal` ===")
    print("  negative = the mirror twin (lighter, grippier); "
          "+1 = the S4 default\n")
    rows, stats = {}, {}
    for k in SWEEP:
        f = TwinFidelity.along("nominal", k * NOMINAL_MAGNITUDE)
        label = f"{k:+.1f}x nominal"
        rows[label] = go(f, label, nl, pl, seeds)
        stats[label] = summarise(label, rows[label])
        stats[label]["k"] = k
        stats[label]["log_error"] = abs(k) * NOMINAL_MAGNITUDE

    print("\n  paired against the exact twin (0.0x):")
    base = "+0.0x nominal"
    comps = {k: paired(rows[base], rows[k], "exact", k)
             for k in rows if k != base}
    return {"stats": stats, "paired": comps, "rows": rows}


def directions(nl, pl, seeds) -> dict:
    print(f"\n=== DIRECTIONS ({REGIME}) — which parameter carries the effect ===")
    rows, stats = {}, {}
    for d in ("mass", "friction", "damping"):
        for sign in (+1, -1):
            f = TwinFidelity.along(d, sign * NOMINAL_MAGNITUDE)
            label = f"{d} {sign:+d}"
            rows[label] = go(f, label, nl, pl, seeds)
            stats[label] = summarise(label, rows[label])
    return {"stats": stats, "rows": rows}


def fixed_plan(nl, pl, seeds) -> dict:
    """Twin fidelity measured properly: one plan, every twin asked about it."""
    print(f"\n=== FIXED PLAN ({REGIME}) — accuracy without plan selection ===")
    print("  every twin predicts the SAME one-step plan; the plant then runs it\n")
    fid = {f"{k:+.1f}": TwinFidelity.along("nominal", k * NOMINAL_MAGNITUDE)
           for k in SWEEP}
    rows: list[dict] = []
    for sd in seeds:
        rows += run_fixed_plan(fid, nl, pl, sd, REGIME)

    stats = {}
    actual = np.mean([r["task_ok"] for r in rows])
    print(f"  {len(rows)} episodes, the fixed plan succeeded {100*actual:.1f}% of the time\n")
    for k in fid:
        agree = np.mean([r["said"][k] == r["task_ok"] for r in rows])
        wrong = np.mean([r["said"][k] and not r["task_ok"] for r in rows])
        miss = np.mean([not r["said"][k] and r["task_ok"] for r in rows])
        optimism = np.mean([r["said"][k] for r in rows])
        stats[k] = {"k": float(k), "agreement": float(agree),
                    "false_confidence": float(wrong), "false_alarm": float(miss),
                    "predicted_clean": float(optimism), "n": len(rows)}
        print(f"  {k:>5}x nominal   agrees {100*agree:5.1f}%   "
              f"said-clean-but-no {100*wrong:5.1f}%   said-no-but-clean {100*miss:5.1f}%   "
              f"predicts clean {100*optimism:5.1f}%")
    return {"stats": stats, "actual_success": float(actual), "n": len(rows)}


# ---------------------------------------------------------------- figure

def plot_sweep(sweep_stats: dict, fixed_stats: dict | None, path: str) -> None:
    """Usefulness and fidelity, measured separately, against the same x-axis.

    The two quantities come from different runs on purpose. *Usefulness* is the
    task success rate when the twin is free to choose and repair its plan.
    *Fidelity* is how often the twin's verdict was right about a plan it did not
    choose -- the fixed one-step plan every twin was asked about, so that plan
    selection cannot leak into the accuracy number.

    They share one y-axis because they are the same unit, per cent of episodes,
    and sharing it is the point: both curves have their minimum at or near the
    *true* parameters, which is not where anyone would put it in advance.

    What the x-axis is *not*: zero means the twin has the plant's true
    parameters, not that it is a perfect twin. Its object positions come from
    the camera and it never sees orientation, at every point on this curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

    pts = sorted(sweep_stats.values(), key=lambda s: s["k"])
    x = [s["k"] for s in pts]
    success = [100 * s["clean"] for s in pts]
    two_step = [100 * s["two_step_share"] for s in pts]
    fixed = None
    if fixed_stats:
        fx = sorted(fixed_stats.values(), key=lambda s: s["k"])
        fixed = ([s["k"] for s in fx], [100 * s["agreement"] for s in fx])

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(8.6, 6.1), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.18})
    fig.patch.set_facecolor(SURFACE)

    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#c3c2b7")
        a.axvline(0, color="#c3c2b7", linewidth=1.5, zorder=1)
        a.tick_params(colors=MUTED, labelsize=9)

    def line(a, xs, ys, colour, label, tag):
        a.plot(xs, ys, "-o", color=colour, linewidth=2, markersize=6, label=label,
               zorder=3, markeredgecolor=SURFACE, markeredgewidth=2)
        a.annotate(tag, (xs[-1], ys[-1]), xytext=(9, 0), textcoords="offset points",
                   color=colour, va="center", fontsize=9.5, fontweight="bold")

    line(ax, x, success, BLUE, "task succeeded (twin chooses the plan)", "useful")
    if fixed:
        line(ax, *fixed, ORANGE, "twin's verdict was right (fixed plan)", "accurate")
    line(bx, x, two_step, AQUA, "chose the two-step plan", "two-step")

    lo = min(success + (fixed[1] if fixed else []))
    hi = max(success + (fixed[1] if fixed else []))
    ax.set_ylim(lo - 4, hi + 4)
    bx.set_ylim(0, max(two_step) + 8)
    ax.set_xticks(x)
    ax.set_xlim(min(x) - 0.4, max(x) + 1.25)

    ax.annotate("twin has the true parameters", (0, hi + 3.2), xytext=(7, 0),
                textcoords="offset points", color=MUTED, fontsize=8.5, va="top")
    bx.set_xlabel("twin mis-specification  (signed, in units of the S4 default)",
                  color=MUTED, fontsize=10, labelpad=8)
    ax.set_ylabel("per cent of episodes", color=MUTED, fontsize=10)
    bx.set_ylabel("per cent", color=MUTED, fontsize=9)
    ax.set_title("The best twin is not the true one",
                 color=INK, fontsize=13.5, loc="left", pad=16)
    leg = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(MUTED)
    fig.text(0.012, 0.012,
             "cluttered regime, 300 episodes per point.  negative = the mirror twin "
             "(lighter, grippier cubes).\nzero is exact PARAMETERS, not a perfect twin: "
             "position always comes from the camera and yaw is never seen.",
             color=MUTED, fontsize=7.8, linespacing=1.6)
    fig.subplots_adjust(left=0.095, right=0.875, top=0.905, bottom=0.175)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    print(f"\nwrote {path}")


# ---------------------------------------------------------------- driver

def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "replot":                       # redraw from finished runs
        with open(os.path.join(ROOT, "results", "s5_sweep.json")) as f:
            sw = json.load(f)["sweep"]["stats"]
        fx = None
        fp = os.path.join(ROOT, "results", "s5_fixedplan.json")
        if os.path.exists(fp):
            with open(fp) as f:
                fx = json.load(f)["fixed_plan"]["stats"]
        plot_sweep(sw, fx, os.path.join(ROOT, "results", "figures", "s5_fidelity.png"))
        return
    nl = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    pl = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2, 3]

    out, path = {}, os.path.join(ROOT, "results", f"s5_{mode}.json")
    if mode in ("factorial", "all"):
        out["factorial"] = factorial(nl, pl, seeds)
    if mode in ("sweep", "all"):
        out["sweep"] = sweep(nl, pl, seeds)
        plot_sweep(out["sweep"]["stats"], None,
                   os.path.join(ROOT, "results", "figures", "s5_fidelity.png"))
    if mode in ("directions", "all"):
        out["directions"] = directions(nl, pl, seeds)
    if mode in ("fixedplan", "all"):
        out["fixed_plan"] = fixed_plan(nl, pl, seeds)

    slim = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in out.items()}
    slim["config"] = {"regime": REGIME, "n_layouts": nl, "per_layout": pl,
                      "seeds": seeds, "nominal_magnitude": NOMINAL_MAGNITUDE}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(slim, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
