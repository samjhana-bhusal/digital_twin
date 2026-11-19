#!/usr/bin/env python
"""Render a pick-and-place episode, and benchmark the skill over random layouts.

    .venv/bin/python experiments/demo_pick_place.py
"""
from __future__ import annotations

import collections
import os
import sys
import time

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fieldtwin import skills
from fieldtwin.arena import build, default_scene, random_scene
from fieldtwin.robot import Panda

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "visuals")
os.makedirs(OUT, exist_ok=True)
W, H = 560, 400


class Recorder:
    """Captures a frame every `every` control steps, from two cameras."""

    def __init__(self, model, cams=("front", "wrist"), every: int = 7):
        self.r = mujoco.Renderer(model, height=H, width=W)
        self.cams, self.every = cams, every
        self.frames: dict[str, list] = {c: [] for c in cams}
        self.n = 0

    def __call__(self, robot: Panda):
        if self.n % self.every == 0:
            for c in self.cams:
                self.r.update_scene(robot.data, camera=c)
                self.frames[c].append(Image.fromarray(self.r.render()))
        self.n += 1


def render_episode() -> None:
    m, sc = build(default_scene())
    d = mujoco.MjData(m)
    robot = Panda(m, d)
    robot.home()
    for _ in range(80):
        robot.step()

    rec = Recorder(m)
    robot.on_step = rec
    pad = sc.landmark("green_pad").pos
    out = skills.pick_and_place(robot, m, d, "red_cube", pad)
    robot.on_step = None
    print(f"  episode: {out.reason}  {out.detail}")

    front, wrist = rec.frames["front"], rec.frames["wrist"]
    combo = []
    for a, b in zip(front, wrist):
        sheet = Image.new("RGB", (W + W // 2, H), (16, 18, 22))
        sheet.paste(a, (0, 0))
        sheet.paste(b.resize((W // 2, H // 2 + 60), Image.LANCZOS), (W, 0))
        combo.append(sheet.resize(((W + W // 2) * 3 // 5, H * 3 // 5), Image.LANCZOS)
                     .quantize(colors=128, dither=Image.NONE))
    path = os.path.join(OUT, "pick_place.gif")
    combo[0].save(path, save_all=True, append_images=combo[1:],
                  duration=110, loop=0, optimize=True)
    print(f"  wrote {path} ({len(combo)} frames, {os.path.getsize(path)/1024:.0f} KB)")

    rr = mujoco.Renderer(m, height=H, width=W)
    m2, sc2 = build(default_scene())
    d2 = mujoco.MjData(m2)
    rb = Panda(m2, d2); rb.home()
    for _ in range(80):
        rb.step()
    rr2 = mujoco.Renderer(m2, height=H, width=W)
    rr2.update_scene(d2, camera="front")
    Image.fromarray(rr2.render()).save(os.path.join(OUT, "arena_front.png"))
    print("  wrote arena_front.png")


def benchmark(n: int = 60) -> None:
    rng = np.random.default_rng(0)
    reasons = collections.Counter()
    errs = []
    t0 = time.time()
    for _ in range(n):
        m, sc = build(random_scene(rng))
        d = mujoco.MjData(m)
        robot = Panda(m, d)
        robot.home()
        for _ in range(80):
            robot.step()
        out = skills.pick_and_place(robot, m, d, "red_cube",
                                    sc.landmark("green_pad").pos)
        reasons[out.reason if out.ok else "FAIL:" + out.reason] += 1
        if out.ok:
            errs.append(out.detail["placement_error_m"])
    ok = sum(v for k, v in reasons.items() if not k.startswith("FAIL"))
    print(f"  {n} layouts in {time.time()-t0:.1f}s -> {ok}/{n} succeeded")
    for k, v in reasons.most_common():
        print(f"    {v:3d}  {k}")
    if errs:
        print(f"    placement error mean {np.mean(errs)*1000:.1f} mm, "
              f"p95 {np.quantile(errs,0.95)*1000:.1f} mm")


if __name__ == "__main__":
    print("rendering episode ...")
    render_episode()
    print("benchmarking ...")
    benchmark()
