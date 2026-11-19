"""Turn results/*.json into LaTeX the paper includes verbatim.

Nothing in paper.tex is a number typed by hand. Every figure in the text is a
macro defined in `generated/numbers.tex` and every table is generated into
`generated/tables.tex`, both produced here from the experiment outputs. The
reason is the obvious one: a paper written by copying numbers out of a terminal
drifts from its own data the first time an experiment is re-run, and the drift is
silent. This way a stale number is a build error, not a reader's problem.

    python paper/build_tables.py

Reads  results/s4_loop.json, s5_factorial.json, s5_sweep.json,
       s5_fixedplan.json, s5_directions.json, s5_identify.json,
       s5_alignment.json
Writes paper/generated/{numbers,tables}.tex
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "generated"

MACROS: dict[str, str] = {}
TABLES: list[str] = []


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval. Preferred over the normal approximation because
    these rates sit near 0.9 with n = 200-300, where the Wald interval
    overcovers below and spills past 1.0 above."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct_ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{100*k/n:.1f} \\ci{{{100*lo:.1f}}}{{{100*hi:.1f}}}"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs, which is the
    conditional-exact form of McNemar's test. Used rather than the chi-square
    approximation because several comparisons here have fewer than ten
    discordant pairs in one direction."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------- plumbing

def key(*parts: str) -> str:
    """A LaTeX-legal macro name: letters only, so digits in arm labels have to
    go. `\\numLoopClutClean` rather than `\\num_loop_cluttered_clean`."""
    s = "".join(p[:1].upper() + p[1:] for p in parts)
    s = re.sub(r"[^A-Za-z]", "", s)
    return "num" + s


def put(name: str, value: str) -> None:
    if name in MACROS and MACROS[name] != value:
        raise SystemExit(f"macro {name} redefined: {MACROS[name]!r} vs {value!r}")
    MACROS[name] = value


def rate(name: str, k: int, n: int) -> float:
    put(name, f"{100*k/n:.1f}")
    lo, hi = wilson(k, n)
    put(name + "Lo", f"{100*lo:.1f}")
    put(name + "Hi", f"{100*hi:.1f}")
    return k / n


def load(stem: str):
    return json.loads((RESULTS / f"{stem}.json").read_text())


def tabular(caption: str, label: str, spec: str, header: list[str],
            rows: list[list[str]], note: str = "", star: bool = False) -> None:
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}",
             "\\small", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\" if r != ["\\midrule"] else "\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\\\[2pt]\\footnotesize\\raggedright {note}")
    lines.append(f"\\end{{{env}}}")
    TABLES.append("\n".join(lines))


# ================================================================ S4
ARM_NAME = {"direct": "direct (no twin)",
            "loop": "loop, mis-specified twin",
            "loop-exact": "loop, exact-physics twin"}
ARM_SHORT = {"direct": "Direct", "loop": "Loop", "loop-exact": "LoopExact"}
REG_SHORT = {"open": "Open", "cluttered": "Clut"}


def s4() -> None:
    d = load("s4_loop")
    cells = {k: v for k, v in d.items() if "/" in k and not k.startswith("paired")}

    # ---- headline table, with per-seed spread and cost -------------------
    rows = []
    for regime in ("open", "cluttered"):
        rows.append(["\\multicolumn{7}{l}{\\itshape " +
                     ("open regime (3 cubes, 85\\,mm minimum separation)" if regime == "open"
                      else "cluttered regime (5 cubes, 62\\,mm minimum separation)") + "}"])
        for arm in ("direct", "loop", "loop-exact"):
            c = cells[f"{arm}/{regime}"]
            rs = c["rows"]
            n = len(rs)
            ok = sum(1 for r in rs if r["task_ok"])
            tag = ARM_SHORT[arm] + REG_SHORT[regime]

            # per-seed spread: the point of F7, so it is in the headline table
            per_seed = defaultdict(lambda: [0, 0])
            for r in rs:
                per_seed[r["seed"]][1] += 1
                per_seed[r["seed"]][0] += bool(r["task_ok"])
            seed_rates = sorted(100 * a / b for a, b in per_seed.values())

            occ = [r for r in rs if r["goal"] == "at-occupied-spot"]
            occ_ok = sum(1 for r in occ if r["task_ok"])
            sims = sum(r["n_sims"] for r in rs) / n
            secs = c["elapsed_s"] / n

            rate(key(tag, "Clean"), ok, n)
            rate(key(tag, "Occ"), occ_ok, len(occ))
            put(key(tag, "Sims"), f"{sims:.2f}")
            put(key(tag, "Secs"), f"{secs:.2f}")
            put(key(tag, "SeedLo"), f"{seed_rates[0]:.1f}")
            put(key(tag, "SeedHi"), f"{seed_rates[-1]:.1f}")
            put(key(tag, "N"), str(n))
            put(key(tag, "NOcc"), str(len(occ)))

            rows.append([ARM_NAME[arm], str(n), pct_ci(ok, n),
                         f"{seed_rates[0]:.1f}--{seed_rates[-1]:.1f}",
                         pct_ci(occ_ok, len(occ)),
                         f"{sims:.2f}", f"{secs:.2f}"])
    tabular(
        "Clean success for the three arms. \\emph{Clean} means the goal predicate holds "
        "\\emph{and} no object the instruction did not license moving was displaced by "
        "more than 30\\,mm. Brackets are Wilson 95\\% intervals; the next column is the "
        "spread of the four per-seed rates, which is wider than several of the effects "
        "this paper measures and is why every comparison in Table~\\ref{tab:paired} is "
        "paired rather than rate-against-rate.",
        "headline",
        "lrlllrr",
        ["arm", "$n$", "clean success (\\%)", "per-seed", "\\emph{where Y is} (\\%)",
         "rollouts", "s/ep."],
        rows,
        star=True)

    # ---- paired comparisons ---------------------------------------------
    prs = [(k, v) for k, v in d.items() if k.startswith("paired/")]
    rows = []
    for k, v in prs:
        _, pair, regime = k.split("/")
        a, _, b = pair.split("-")
        lbl = {"loop": "loop", "exact": "loop, exact-physics", "direct": "direct"}
        rows.append([f"{lbl.get(a,a)} vs.\\ {lbl.get(b,b)}", regime, str(v["n"]),
                     str(v["rescued"]), str(v["broke"]),
                     signed(v["rescued"] - v["broke"]), mathp(v["p"])])
        tag = key("Paired", a, b, regime)
        put(tag + "Resc", str(v["rescued"]))
        put(tag + "Broke", str(v["broke"]))
        put(tag + "P", fmt_p(v["p"]))
    tabular(
        "Paired comparisons. Every arm ran the same layouts with the same instructions "
        "under the same seeds, so each episode is a matched pair and the comparison "
        "reduces to the episodes where the two arms disagreed. \\emph{rescued} is "
        "episodes the first arm cleaned that the second failed; $p$ is McNemar's exact "
        "test on the discordant pairs.",
        "paired", "llrrrrl",
        ["comparison", "regime", "$n$", "rescued", "broke", "net", "$p$"],
        rows)

    # ---- who can tell a failure: verifier vs the skill's own report ------
    pooled = defaultdict(Counter)
    for name, c in cells.items():
        arm = name.split("/")[0]
        for r in c["rows"]:
            truth = bool(r["task_ok"])
            pooled["camera"][(r["verdict"] == "success", truth)] += 1
            pooled["skill"][(bool(r["skill_ok"]), truth)] += 1
            if arm != "direct":
                pooled["twin"][(bool(r["twin_said_clean"]), truth)] += 1
    rows = []
    src_name = {"skill": "the skill, from inside the action",
                "twin": "the twin, before acting",
                "camera": "the camera, from one photograph"}
    for src in ("skill", "twin", "camera"):
        c = pooled[src]
        tp, fp = c[(True, True)], c[(True, False)]
        fn, tn = c[(False, True)], c[(False, False)]
        said = tp + fp
        prec = tp / said if said else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        rows.append([src_name[src], group(said), str(fp), pct_ci(tp, said),
                     f"{100*rec:.1f}", str(fn)])
        t = key("Judge", src)
        put(t + "Said", group(said)); put(t + "Wrong", str(fp))
        put(t + "Prec", f"{100*prec:.1f}"); put(t + "Miss", str(fn))
        lo, hi = wilson(tp, said); put(t + "PrecLo", f"{100*lo:.1f}")
    tabular(
        "Three judges of the same 1{,}800 episodes, scored against the simulator's own "
        "goal predicate. The skill knows only that it grasped, carried and released; "
        "the twin predicts before acting; the camera looks afterwards. Only the "
        "false-positive column matters operationally --- it is the error a system acts "
        "on, and it is the only column that separates them: recall is near-identical "
        "because all three judges rarely reject an episode that was in fact clean. "
        "The twin's row covers the 1{,}200 episodes of the two loop arms.",
        "judges", "lrrlrr",
        ["judge", "success calls", "wrong", "precision (\\%)", "recall (\\%)",
         "missed"],
        rows)

    # ---- failure taxonomy -----------------------------------------------
    tax = defaultdict(Counter)
    for name, c in cells.items():
        arm, regime = name.split("/")
        for r in c["rows"]:
            if r["task_ok"]:
                continue
            if not r["goal_ok"]:
                cause = r["reason"] if not r["skill_ok"] else "goal_unmet_silently"
            else:
                cause = "collateral_damage"
            tax[(arm, regime)][cause] += 1
    causes = sorted({c for v in tax.values() for c in v}, key=lambda c: -sum(
        v[c] for v in tax.values()))
    pretty = {"goal_unmet_silently": "skill succeeded, goal unmet",
              "collateral_damage": "goal met, scene disturbed $>$30\\,mm",
              "grasp_missed": "fingers closed on nothing",
              "placed_off_target": "released $>$50\\,mm from target",
              "dropped_in_transit": "lost the object carrying it",
              "object_slipped": "slipped out on the lift",
              "lost_on_lift": "rode up wedged, then fell",
              "approach_unreachable": "IK failed on approach",
              "descend_unreachable": "IK failed descending",
              "place_unreachable": "IK failed at the target",
              "transit_unreachable": "IK failed in transit",
              "no_plan": "no legal placement found"}
    rows = []
    for c in causes:
        rows.append([pretty.get(c, c.replace("_", "\\_"))] +
                    [str(tax[(a, g)][c] or "\\textendash")
                     for g in ("open", "cluttered")
                     for a in ("direct", "loop", "loop-exact")])
    rows.append(["\\midrule"])
    rows.append(["\\textbf{total failures}"] +
                [f"\\textbf{{{sum(tax[(a,g)].values())}}}"
                 for g in ("open", "cluttered")
                 for a in ("direct", "loop", "loop-exact")])
    put(key("TaxSilentTotal"), str(sum(v["goal_unmet_silently"] for v in tax.values())))
    put(key("TaxCollateralTotal"), str(sum(v["collateral_damage"] for v in tax.values())))
    tabular(
        "Failure taxonomy over all 1{,}800 episodes, by first cause. The two rows the "
        "loop was built to attack are the top two: an episode where the skill reported "
        "success and the goal was not met, and one where the goal was met by disturbing "
        "protected objects. Rows below them are ordinary skill failures, which the loop "
        "repairs only incidentally.",
        "taxonomy", "l" + "r" * 6,
        ["first cause", "\\multicolumn{3}{c}{open}",
         "\\multicolumn{3}{c}{cluttered}"],
        [["", "dir.", "loop", "exact", "dir.", "loop", "exact"]] + rows,
        note="dir.\\ = direct, exact = loop with the exact-physics twin.")

    # ---- selective autonomy: what the unendorsed flag is worth -----------
    rows = []
    for regime in ("open", "cluttered"):
        for arm in ("loop", "loop-exact"):
            rs = cells[f"{arm}/{regime}"]["rows"]
            end = [r for r in rs if r["twin_ok"]]
            un = [r for r in rs if not r["twin_ok"]]
            e_ok = sum(1 for r in end if r["task_ok"])
            u_ok = sum(1 for r in un if r["task_ok"])
            tag = key("Sel", ARM_SHORT[arm], REG_SHORT[regime])
            put(tag + "Cov", f"{100*len(end)/len(rs):.1f}")
            put(tag + "NUn", str(len(un)))
            rate(tag + "End", e_ok, len(end))
            if un:
                rate(tag + "Un", u_ok, len(un))
            rows.append([ARM_NAME[arm], regime,
                         f"{100*len(end)/len(rs):.1f}",
                         pct_ci(e_ok, len(end)),
                         str(len(un)),
                         pct_ci(u_ok, len(un)) if un else "\\textendash"])
    tabular(
        "Selective autonomy, for free. A plan the twin never endorsed within its budget "
        "is still executed here, but flagged. The flag is the loop's own estimate that "
        "it is about to do something it could not verify, and episodes carrying it fail "
        "far more often --- which is what makes it usable as a request for help rather "
        "than as a label after the fact.",
        "selective", "llrlrl",
        ["arm", "regime", "coverage (\\%)", "clean $\\mid$ endorsed (\\%)",
         "$n$ flagged", "clean $\\mid$ flagged (\\%)"],
        rows)

    # pooled version of the same, which is the number quoted in the text
    end = [r for n, c in cells.items() if not n.startswith("direct")
           for r in c["rows"] if r["twin_ok"]]
    un = [r for n, c in cells.items() if not n.startswith("direct")
          for r in c["rows"] if not r["twin_ok"]]
    rate(key("SelPoolEnd"), sum(1 for r in end if r["task_ok"]), len(end))
    rate(key("SelPoolUn"), sum(1 for r in un if r["task_ok"]), len(un))
    put(key("SelPoolNUn"), str(len(un)))
    put(key("SelPoolCov"), f"{100*len(end)/(len(end)+len(un)):.1f}")

    collateral_sensitivity(cells)

    # ---- which repair, and did it work ---------------------------------
    rows = []
    for regime in ("open", "cluttered"):
        rs = cells[f"loop/{regime}"]["rows"]
        by = defaultdict(lambda: [0, 0])
        for r in rs:
            by[r["plan_label"]][1] += 1
            by[r["plan_label"]][0] += bool(r["task_ok"])
        for lbl, (ok, n) in sorted(by.items(), key=lambda kv: -kv[1][1]):
            rows.append([lbl, regime, str(n), f"{100*n/len(rs):.1f}", pct_ci(ok, n)])
            t = key("Repair", lbl.replace("-", ""), REG_SHORT[regime])
            put(t + "N", str(n)); rate(t, ok, n)
    tabular(
        "Which plan the loop executed, and how it fared. \\texttt{direct} is the "
        "one-step plan the baseline would also have run; the rest are repairs the twin "
        "demanded. \\texttt{clear-then-place} is the only repair that changes the "
        "\\emph{length} of the plan, and it carries the effect.",
        "repairs", "llrrl",
        ["plan executed", "regime", "$n$", "share (\\%)", "clean success (\\%)"],
        rows)


WORD = {0.5: "Half", 1.0: "One", 2.0: "Two", 3.0: "Three"}


def kname(k: float) -> str:
    """Macro-safe name for a signed sweep point: digits are not legal in a
    LaTeX macro name, so -0.5 becomes `MinusHalf`."""
    if k == 0.0:
        return "Zero"
    return ("Minus" if k < 0 else "Plus") + WORD[abs(k)]


def collateral_sensitivity(cells) -> None:
    r"""Does the headline result depend on where the collateral line is drawn?

    It has to be asked, because in these 1,800 episodes the 30 mm rule never
    binds: no episode met its goal and failed on displacement, and the largest
    displacement observed sits just under the threshold. A scoring constant that
    never fires in the experiment reporting it is untested, so the honest thing
    is to re-score every episode at tighter thresholds and show whether the
    paired comparison survives.
    """
    taus = ((0.010, "Ten"), (0.015, "Fifteen"), (0.020, "Twenty"),
            (0.030, "Thirty"), (0.050, "Fifty"))
    worst = max(r["collateral_m"] for c in cells.values() for r in c["rows"])
    put(key("CollateralMax"), f"{1000*worst:.1f}")
    rows = []
    for tau, word in taus:
        line = [f"{1000*tau:.0f}"]
        for regime in ("open", "cluttered"):
            scored = {}
            for arm in ("direct", "loop"):
                rs = cells[f"{arm}/{regime}"]["rows"]
                scored[arm] = [bool(r["goal_ok"]) and r["collateral_m"] < tau
                               for r in rs]
            a, b = scored["loop"], scored["direct"]
            resc = sum(1 for x, y in zip(a, b) if x and not y)
            broke = sum(1 for x, y in zip(a, b) if y and not x)
            pv = mcnemar_exact(resc, broke)
            line += [f"{100*sum(b)/len(b):.1f}", f"{100*sum(a)/len(a):.1f}",
                     f"{resc:+d}/$-${broke}", mathp(pv)]
            t = key("Tau", word, REG_SHORT[regime])
            put(t + "Direct", f"{100*sum(b)/len(b):.1f}")
            put(t + "Loop", f"{100*sum(a)/len(a):.1f}")
            put(t + "Resc", str(resc))
            put(t + "Broke", str(broke))
            put(t + "P", fmt_p(pv))
        rows.append(line)
    tabular(
        "Re-scoring every episode at tighter collateral thresholds. The 30\\,mm rule "
        "used throughout never binds in these runs --- no episode met its goal and "
        "failed on displacement, and the largest displacement seen anywhere was "
        "\\numCollateralMax\\,mm --- so the rule as stated is untested by this "
        "experiment. Tightening it to 10\\,mm costs both arms several points and "
        "leaves the paired result intact, which is the claim that needed checking.",
        "collateral", "r" + "rrll" * 2,
        ["$\\tau$ (mm)", "\\multicolumn{4}{c}{open}",
         "\\multicolumn{4}{c}{cluttered}"],
        [["", "direct", "loop", "net", "$p$", "direct", "loop", "net", "$p$"]] + rows,
        note="$\\tau$ is the displacement a protected object is allowed. Rates are clean "
             "success (\\%); \\emph{net} is rescued/broke for loop against direct, paired "
             "per episode.",
        star=True)


def fmt_p(p: float) -> str:
    if p < 1e-4:
        e = math.floor(math.log10(p))
        return f"{p/10**e:.0f}\\times 10^{{{e}}}"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def group(n: int) -> str:
    """Thousands separator as LaTeX wants it, so 1{,}693 sits beside 1{,}800
    in the same sentence rather than reading as a different kind of number."""
    return f"{n:,}".replace(",", "{,}")


def signed(n: int) -> str:
    """A signed integer for a table cell, with a real minus rather than the
    hyphen `{:+d}` produces outside math mode."""
    return f"$-{abs(n)}$" if n < 0 else f"$+{n}$"


def mathp(p: float) -> str:
    """The same value wrapped for a table cell outside math mode. The macro
    form stays bare so prose can write `$p = \\numFoo$` without nesting."""
    return "$" + fmt_p(p) + "$"


# ================================================================ S5
def s5() -> None:
    fac = load("s5_factorial")["factorial"]
    rows = []
    order = ["direct (no twin)", "exact physics, yaw unseen", "exact physics, yaw known",
             "mis-specified, yaw unseen", "mis-specified, yaw known"]
    tags = {"direct (no twin)": "FacNone",
            "exact physics, yaw unseen": "FacExactBlind",
            "exact physics, yaw known": "FacExactSeen",
            "mis-specified, yaw unseen": "FacWrongBlind",
            "mis-specified, yaw known": "FacWrongSeen"}
    for lbl in order:
        s = fac["stats"][lbl]
        n = s["n"]
        k = round(s["clean"] * n)
        rows.append([lbl, str(n), pct_ci(k, n),
                     f"{100*s['occupied_spot']:.1f}",
                     "\\textendash" if lbl.startswith("direct") else f"{100*s['twin_agreement']:.1f}",
                     "\\textendash" if lbl.startswith("direct") else f"{100*s['twin_false_confidence']:.1f}",
                     f"{100*s['two_step_share']:.1f}"])
        t = key(tags[lbl])
        rate(t, k, n)
        put(t + "Agree", "--" if lbl.startswith("direct") else f"{100*s['twin_agreement']:.1f}")
        put(t + "False", "--" if lbl.startswith("direct") else f"{100*s['twin_false_confidence']:.1f}")
        put(t + "TwoStep", f"{100*s['two_step_share']:.1f}")
    tabular(
        "Crossing the twin's two kinds of ignorance: wrong physics, and blindness to "
        "which way a cube is turned. The two are \\emph{substitutes} --- fixing either "
        "lifts the worst cell about five points, and fixing both lifts it the same five "
        "points and no more. \\emph{false confidence} is the share of episodes the twin "
        "predicted clean and were not, the error the system acts on.",
        "factorial", "lrlrrrr",
        ["twin", "$n$", "clean success (\\%)", "\\emph{where Y is}",
         "twin agreed", "false conf.", "two-step"],
        rows,
        note="Cluttered regime throughout, 300 episodes per cell, identical layouts and "
             "instructions across cells. Paired tests against the worst cell are in the "
             "text.")

    # paired, factorial
    for k2, v in fac["paired"].items():
        t = key("FacPaired", re.sub(r"[^a-z]", "", k2.lower())[:18])
        put(t + "Resc", str(v["rescued"])); put(t + "Broke", str(v["broke"]))
        put(t + "P", fmt_p(v["p"]))

    # ---- the sweep, with the fixed-plan column beside it ----------------
    sw = load("s5_sweep")["sweep"]
    fp = load("s5_fixedplan")["fixed_plan"]
    put(key("FixedActual"), f"{100*fp['actual_success']:.1f}")
    put(key("FixedN"), str(fp["n"]))
    rows = []
    for lbl, s in sw["stats"].items():
        k = s["k"]
        f = fp["stats"][f"{k:+.1f}".replace("+0.0", "+0.0")]
        n = s["n"]
        ok = round(s["clean"] * n)
        ag = round(f["agreement"] * n)
        pr = sw["paired"].get(lbl)
        rows.append([f"${k:+.1f}$", f"{s['log_error']:.2f}",
                     pct_ci(ok, n), f"{100*s['occupied_spot']:.1f}",
                     pct_ci(ag, n), f"{100*f['false_confidence']:.1f}",
                     f"{100*f['predicted_clean']:.1f}",
                     "\\textendash" if pr is None else
                     f"{pr['rescued']:+d}/$-${pr['broke']} (${fmt_p(pr['p'])}$)"])
        t = key("Sweep", kname(k))
        rate(t, ok, n); rate(t + "Acc", ag, n)
        put(t + "False", f"{100*f['false_confidence']:.1f}")
        put(t + "Pred", f"{100*f['predicted_clean']:.1f}")
        if pr:
            put(t + "Resc", str(pr["rescued"])); put(t + "Broke", str(pr["broke"]))
            put(t + "P", fmt_p(pr["p"]))
    tabular(
        "Mis-specification swept through zero as a \\emph{signed} magnitude along one "
        "direction in log-parameter space, so that $-1$ is the exact mirror of $+1$: "
        "where the positive twin believes cubes are heavier and more slippery than they "
        "are, the negative one believes they are lighter and grippier. \\emph{useful} and "
        "\\emph{accurate} come from two separate runs on purpose --- accuracy is measured "
        "on one fixed plan that no twin chose, because a twin scored on the plan it "
        "selected is being scored on its own caution.",
        "sweep", "rrlrlrrl",
        ["$k$", "$\\lVert e\\rVert$", "useful: clean (\\%)", "\\emph{where Y is}",
         "accurate (\\%)", "false conf.", "pred.\\ clean", "paired vs.\\ $k{=}0$"],
        rows,
        note="$k$ is in units of the mis-specification the loop was built with; "
             "$\\lVert e\\rVert$ is the resulting distance from the truth in log-parameter "
             "space. The fixed plan actually succeeded \\numFixedActual\\% of the time, so "
             "a twin predicting \\emph{clean} far above that figure is right by agreeing "
             "with the common case rather than by discriminating.",
        star=True)

    # ---- identification against task sensitivity ------------------------
    al = load("s5_alignment")
    idf = load("s5_identify")
    dirs = load("s5_directions")["directions"]["stats"]
    pretty = {"cube_mass": "cube mass", "cube_friction": "cube friction",
              "joint_damping": "joint damping"}
    rows = []
    for p in ("cube_friction", "cube_mass", "joint_damping"):
        sig = al["identification_sigma_per_log"][p]
        tsk = al["task_sensitivity_pts_per_log"][p]
        up = dirs[f"{p.split('_')[1] if p!='joint_damping' else 'damping'} +1"]["clean"]
        dn = dirs[f"{p.split('_')[1] if p!='joint_damping' else 'damping'} -1"]["clean"]
        rows.append([pretty[p], f"{sig:.1f}", f"{tsk:.1f}",
                     f"{100*up:.1f}", f"{100*dn:.1f}"])
        t = key("Align", p.replace("_", ""))
        put(t + "Sigma", f"{sig:.1f}"); put(t + "Task", f"{tsk:.1f}")
        put(t + "Up", f"{100*up:.1f}"); put(t + "Down", f"{100*dn:.1f}")
    put(key("AlignCos"), f"{al['cosine']:.3f}")
    put(key("AlignAngle"), f"{al['angle_deg']:.1f}")
    put(key("IdentRank"), ", ".join(pretty[p] for p in idf["identification_rank"]))
    put(key("SigmaCube"), f"{1000*idf['sigma']['cube_m']:.1f}")
    put(key("SigmaJoint"), f"{1000*idf['sigma']['joint_rad']:.1f}")
    tabular(
        "What the data pins down against what the task can feel, over the same three "
        "parameters in the same units of log-parameter error. The top two are swapped: "
        "a push constrains friction roughly three times better than mass, while the "
        "task is half again more sensitive to mass than to friction. Joint damping is "
        "the sharper case --- comfortably identifiable, and worth nothing.",
        "alignment", "lrrrr",
        ["parameter", "identification ($\\sigma$ / unit)",
         "task (points / unit)", "clean at $+1$", "clean at $-1$"],
        rows,
        note="Identification sensitivity is the residual change per unit log-parameter "
             "error, referenced to sensor noise ($\\numSigmaCube$\\,mm on cube position, "
             "$\\numSigmaJoint$\\,mrad on joint angle). As unit vectors the two columns sit "
             "$\\numAlignAngle^\\circ$ apart (cosine $\\numAlignCos$).")


# ================================================================ main
def main() -> None:
    s4()
    s5()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.tex").write_text(
        "% generated by paper/build_tables.py -- do not edit\n" +
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(MACROS.items())))
    (OUT / "tables.tex").write_text(
        "% generated by paper/build_tables.py -- do not edit\n\n" +
        "\n\n".join(TABLES) + "\n")
    print(f"wrote {len(MACROS)} macros and {len(TABLES)} tables to {OUT}")


if __name__ == "__main__":
    main()
