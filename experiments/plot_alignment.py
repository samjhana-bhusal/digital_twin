"""The figure for F11: what the data pins down against what the task can feel.

Two panels rather than one, deliberately. The two quantities have different
units -- multiples of sensor noise on one side, percentage points of clean
success on the other -- and there is no honest way to draw them against a shared
axis. Small multiples with the same category order on both sides say the same
thing without inviting the reader to compare bar lengths across the gap: what
matters is the *ordering*, and the ordering is not the same.

    python experiments/plot_alignment.py        ->  results/figures/s5_alignment.png
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
PRETTY = {"cube_mass": "cube mass", "cube_friction": "cube friction",
          "joint_damping": "joint damping"}


def main(path: str | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    al = json.loads((ROOT / "results" / "s5_alignment.json").read_text())
    ident = al["identification_sigma_per_log"]
    task = al["task_sensitivity_pts_per_log"]

    # One shared ordering for both panels, taken from the identification side,
    # so the task panel visibly fails to agree with it.
    order = sorted(ident, key=lambda p: -ident[p])
    y = list(range(len(order)))[::-1]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 2.9), dpi=200,
                                 gridspec_kw={"wspace": 0.46})
    fig.patch.set_facecolor(SURFACE)

    panels = [
        (ax, BLUE, ident, "what the data pins down",
         "identification sensitivity\n($\\sigma$ of residual per unit log error)"),
        (bx, ORANGE, task, "what the task can feel",
         "task sensitivity\n(points of clean success per unit log error)"),
    ]
    for a, colour, values, title, xlabel in panels:
        a.set_facecolor(SURFACE)
        a.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color("#c3c2b7")
        a.tick_params(colors=MUTED, labelsize=8.5, length=0)

        vals = [values[p] for p in order]
        a.barh(y, vals, height=0.56, color=colour, zorder=3)
        a.set_yticks(y)
        a.set_yticklabels([PRETTY[p] for p in order], color=INK, fontsize=9.5)
        a.set_xlim(0, max(vals) * 1.30 if max(vals) else 1)

        rank = {p: i + 1 for i, p in enumerate(sorted(values, key=lambda q: -values[q]))}
        wide = max(vals) * 0.14
        for yy, p, v in zip(y, order, vals):
            # A bar too short to hold its own rank label gets it after the
            # value instead; the zero bar in the task panel is the whole reason
            # this branch exists.
            tag = f"{v:.1f}" if v > wide else f"{v:.1f}   #{rank[p]}"
            a.annotate(tag, (v, yy), xytext=(7, 0), fontsize=9,
                       textcoords="offset points", va="center",
                       color=colour if v > wide else MUTED, fontweight="bold")
            if v > wide:
                a.annotate(f"#{rank[p]}", (0, yy), xytext=(7, 0), fontsize=8.5,
                           textcoords="offset points", va="center",
                           color=SURFACE, fontweight="bold", zorder=4)
        a.set_title(title, color=INK, fontsize=10.5, fontweight="bold",
                    loc="left", pad=9)
        a.set_xlabel(xlabel, color=MUTED, fontsize=8.5)

    fig.suptitle("The parameter the data pins down is not the parameter the task cares about",
                 color=INK, fontsize=12, fontweight="bold", x=0.012, ha="left", y=1.10)
    fig.text(0.012, -0.20,
             f"Ranks (#) are within each panel. The top two are swapped: a push "
             f"constrains friction about three times better than mass, while the task is "
             f"half again more\nsensitive to mass than to friction. As unit vectors the "
             f"two sensitivities sit {al['angle_deg']:.1f}$^\\circ$ apart "
             f"(cosine {al['cosine']:.2f}). Joint damping is comfortably identifiable and "
             f"worth nothing at all.",
             color=MUTED, fontsize=8, ha="left", va="top")

    out = path or str(ROOT / "results" / "figures" / "s5_alignment.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.42)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
