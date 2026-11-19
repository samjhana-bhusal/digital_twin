#!/usr/bin/env python
"""One episode, twice: without the loop and with it.

    .venv/bin/python experiments/demo_loop.py [seed]

Produces `results/figures/s4_loop.png` -- a strip of what the robot's own
overhead camera sees. Top row is the direct attempt, which drops the cube onto
an occupied spot and leaves it perched off target. Bottom row is the same task
after the twin has rejected that plan: the blocker is moved first, and the
placement lands.

The point of the figure is that both rows are the *robot's* view, not a
privileged one. Everything the loop knows, it got from these images.
"""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fieldtwin import plan as planning
from fieldtwin import tasks, verify
from fieldtwin.arena import build, random_scene
from fieldtwin.perceive import Perception, StateView
from fieldtwin.robot import ARM_DOF, Panda
from fieldtwin.twin import Choice, Twin, TwinFidelity, choose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE = (252, 252, 251)
INK = (11, 11, 11)
DIM = (137, 135, 129)

_FONTS = ("/System/Library/Fonts/Supplemental/Arial.ttf",
          "/System/Library/Fonts/Helvetica.ttc",
          "/Library/Fonts/Arial.ttf")


def _font(size: int = 15):
    for path in _FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def find_episode(seed: int):
    """A layout holding an occupied-destination task -- the class the loop is for."""
    for li in range(60):
        rng = np.random.default_rng([seed, li])
        m, sc = build(random_scene(rng, n_cubes=4, min_sep=0.070))
        d = mujoco.MjData(m)
        robot = Panda(m, d)
        robot.home()
        for _ in range(80):
            robot.step()
        rest = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())
        for t in tasks.enumerate_tasks(m, d, sc, rng):
            if isinstance(t.goal, tasks.AtObjectSpotGoal):
                return m, d, robot, sc, rest, t, rng
    raise RuntimeError("no occupied-destination task found")


def attempt(m, d, robot, sc, rest, task, eye, use_twin: bool, rng):
    d.qpos[:], d.qvel[:], d.ctrl[:] = rest
    mujoco.mj_forward(m, d)
    before = verify.look(eye, robot, d, sc, then_ready=True)
    shots = [eye.image(d)]
    view = StateView(before)
    truth_start = {o.name: tasks.obj_xy(d, o.name).copy() for o in sc.objects}

    if use_twin:
        twin = Twin(sc, TwinFidelity())
        choice = choose(task, m, view, sc, twin, rng, arm_q=robot.q_arm,
                        ctrl=d.ctrl.copy(),
                        gripper_q=d.qpos[ARM_DOF:ARM_DOF + 2].copy())
    else:
        choice = Choice(planning.direct_plan(task, m, view, sc, rng), None, [], False)

    for step in choice.plan.steps:
        step.run(robot, m, d)
        robot.park_for_view()
        shots.append(eye.image(d))

    goal_ok = bool(task.satisfied(m, d, sc))
    damage = task.collateral(m, d, sc, truth_start)
    after = verify.look(eye, robot, d, sc)
    verdict = verify.verify(task, sc, before, after)
    return shots, choice, goal_ok, damage, verdict


def _wrap(dr, text, font, limit):
    """Greedy wrap to `limit` pixels. Captions used to be drawn as one line at
    the tile's left edge, which ran the longest of them past the panel and off
    the canvas -- the right-hand caption in the published figure was clipped
    mid-word. Two short lines fit; one long one does not."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if dr.textlength(trial, font=font) <= limit or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:2]


def strip(shots, captions, width=320, slots=None):
    tiles = [Image.fromarray(s).resize((width, width), Image.LANCZOS) for s in shots]
    slots = slots or len(tiles)
    pad, head, lead = 12, 46, 19
    out = Image.new("RGB", (slots * width + (slots + 1) * pad,
                            width + head + 2 * pad), SURFACE)
    dr = ImageDraw.Draw(out)
    font = _font(14)
    for i, (tile, cap) in enumerate(zip(tiles, captions)):
        x = pad + i * (width + pad)
        out.paste(tile, (x, head + pad))
        for j, line in enumerate(_wrap(dr, cap, font, width - 4)):
            dr.text((x + 2, 8 + j * lead), line, fill=INK if i else DIM, font=font)
    return out


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    m, d, robot, sc, rest, task, rng = find_episode(seed)
    eye = Perception(m)
    print(f"instruction: {task.instruction}")

    rows = []
    for use_twin, name in ((False, "no twin"), (True, "simulate first")):
        shots, choice, goal_ok, damage, verdict = attempt(
            m, d, robot, sc, rest, task, eye,
            use_twin, np.random.default_rng([seed, int(use_twin)]))
        caps = [f"{name}: as seen before acting"]
        for s in choice.plan.steps:
            caps.append(f"after: {s.describe().split(' [')[0]}")
        caps[-1] += f" -- goal {'reached' if goal_ok else 'MISSED'}"
        print(f"  [{name:14s}] plan={choice.plan.label:16s} sims={choice.n_sims} "
              f"goal={goal_ok} damage={damage*1000:.0f}mm camera={verdict.label()}")
        rows.append((shots, caps))

    # Both rows get the same number of tile slots so the two strips are the
    # same width; otherwise the shorter row's captions are clipped at the edge
    # of its own canvas.
    slots = max(len(shots) for shots, _ in rows)
    rows = [strip(shots, caps, slots=slots) for shots, caps in rows]
    w = max(r.width for r in rows)
    fig = Image.new("RGB", (w, sum(r.height for r in rows)), SURFACE)
    y = 0
    for r in rows:
        fig.paste(r, (0, y))
        y += r.height
    out = os.path.join(ROOT, "results", "figures", "s4_loop.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.save(out)
    eye.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
