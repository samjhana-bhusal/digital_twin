#!/usr/bin/env python
"""Generate the explainer's figures and data.

Nothing here is illustrative. Every number the explainer page shows is produced
by this script, including the caveat it reports about relative eigenvalue
thresholds.

Two demonstrations:

  A. Identifiability, cleanly.  Spectrum from PUSH-ONLY data. The flat subspace
     is one-dimensional (`mu_tip`, exactly unidentifiable because MuJoCo takes
     the elementwise max of contact frictions). A twin wrong by 57% along it
     behaves identically to the plant; a twin wrong by the same amount in an
     identifiable direction diverges visibly. This is what identifiability means,
     with no interpretation required.

  B. The threshold artefact.  Spectrum from MIXED data (push + contact-free
     chirps). Free-space chirps identify the arm parameters so sharply that the
     largest eigenvalue grows by orders of magnitude, which pushes every contact
     parameter below a RELATIVE cutoff and labels it unidentifiable even though
     the push episodes constrain it perfectly well. Sweeping the perturbation
     magnitude shows the local Hessian's prediction degrading and then inverting.
"""
from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fieldtwin.bus import BusConfig
from fieldtwin.identify.loss import CHANNELS, fitting_twin, noise_sigmas, residuals
from fieldtwin.identify.spectrum import finite_difference_jacobian, spectrum_from_jacobian
from fieldtwin.plant import ARM_DOF, World, make_plant
from fieldtwin.rollout import make_dataset, plan_tip_line

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "visuals")
os.makedirs(OUT, exist_ok=True)

W, H = 520, 340
PLANT_BUS = BusConfig(seed=7)
CLEAN = PLANT_BUS.deterministic()
TOL = 1e-3
MAG = 0.45                     # log-space perturbation: a 57% parameter error
SIG = noise_sigmas(PLANT_BUS)


def run_logged(world: World, cmd: np.ndarray, box0=None) -> dict:
    obs = world.reset(q_arm=cmd[0], box=box0)
    rec: dict[str, list] = {k: [] for k in ("t", "cmd", "eff", "q", "v", "qtrue", "box", "tip")}
    for k in range(cmd.shape[0]):
        rec["t"].append(obs.t)
        rec["cmd"].append(cmd[k].copy())
        rec["eff"].append(world.bus.effective_ctrl(world.data.time).copy())
        rec["q"].append(obs.q_arm.copy())
        rec["v"].append(obs.v_arm.copy())
        rec["qtrue"].append(world.data.qpos[:ARM_DOF].copy())
        rec["box"].append(obs.box.copy())
        rec["tip"].append(obs.tip.copy())
        obs = world.step_control(cmd[k])
    return {k: np.asarray(v) for k, v in rec.items()}


def observed(r: dict) -> np.ndarray:
    """The same 11 channels, in the same order, that the loss uses."""
    return np.concatenate([r["q"], r["v"], r["box"], r["tip"]], axis=1)


def push_cmd(world: World, approach: float, offset_frac: float, travel: float, n: int = 110):
    centre = world.model.body("box").pos[:2].copy()
    half = float(world.model.geom("g_box").size[0])
    u = np.array([np.cos(approach), np.sin(approach)])
    perp = np.array([-u[1], u[0]])
    off = offset_frac * half
    return plan_tip_line(world, centre - u * (half + 0.030) + perp * off,
                         centre + u * travel + perp * off, n)


def spectrum_for(trajs, twin, space):
    x0 = space.to_log(space.nominal)
    J = finite_difference_jacobian(
        lambda xl: residuals(space.from_log(xl), trajs, twin, SIG), x0, eps=1e-3)
    return spectrum_from_jacobian(J, space.names)


def label(img: Image.Image, text: str):
    im = img.copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(16, 18, 22))
    d.text((8, 6), text, fill=(236, 238, 242))
    return im


def render_frames(world: World, cmd: np.ndarray, cam: str, stride: int = 2):
    r = mujoco.Renderer(world.model, height=H, width=W)
    world.reset(q_arm=cmd[0])
    frames = []
    for k in range(cmd.shape[0]):
        if k % stride == 0:
            r.update_scene(world.data, camera=cam)
            frames.append(Image.fromarray(r.render()))
        world.step_control(cmd[k])
    return frames


def main() -> None:
    plant = make_plant(bus_cfg=CLEAN)
    twin = fitting_twin(data_bus=CLEAN)
    space = twin.space
    x0 = space.to_log(space.nominal)
    rng = np.random.default_rng(5)

    def unit_in(basis):
        c = rng.standard_normal(basis.shape[1])
        v = basis @ c
        return v / np.linalg.norm(v)

    cmd_eval = push_cmd(plant, approach=0.35, offset_frac=0.30, travel=0.075)
    cmd_novel = push_cmd(plant, approach=2.45, offset_frac=-0.75, travel=0.055)

    def divergence(theta, cmd):
        pl = make_plant(bus_cfg=CLEAN)
        tw = World(theta=theta, bus_cfg=CLEAN, space=space)
        a, b = run_logged(tw, cmd), run_logged(pl, cmd)
        dw = (observed(a) - observed(b)) / SIG[None, :]
        box = np.linalg.norm(a["box"][:, :2] - b["box"][:, :2], axis=1)
        return {"whitened_rms": float(np.sqrt((dw ** 2).mean())),
                "final_box_mm": float(box[-1] * 1000),
                "max_box_mm": float(box.max() * 1000)}

    # ---------------- A: push-only, where identifiability is unambiguous ----------------
    print("== A: push-only spectrum ==")
    trajs_push = make_dataset(plant, n_push=30, n_free=0, seed=1, n_steps=110)
    Sp = spectrum_for(trajs_push, twin, space)
    st_p, fl_p = Sp.stiff_subspace(TOL), Sp.flat_subspace(TOL)
    print(f"  eff rank {Sp.effective_rank(TOL)}/18, null {Sp.null_dimension()}, "
          f"stiff {st_p.shape[1]}, flat {fl_p.shape[1]}")
    d_stiff = MAG * unit_in(st_p)
    d_flat = MAG * unit_in(fl_p)
    th_stiff = space.from_log(x0 + d_stiff)
    th_flat = space.from_log(x0 + d_flat)
    demoA = {
        "familiar": {"stiff": divergence(th_stiff, cmd_eval), "flat": divergence(th_flat, cmd_eval)},
        "novel": {"stiff": divergence(th_stiff, cmd_novel), "flat": divergence(th_flat, cmd_novel)},
    }
    for task, v in demoA.items():
        print(f"  {task:9s} stiff {v['stiff']['whitened_rms']:8.3f}  "
              f"flat {v['flat']['whitened_rms']:8.3f}  "
              f"(box {v['stiff']['final_box_mm']:.2f} / {v['flat']['final_box_mm']:.2f} mm)")

    # ---------------- B: mixed data, and the threshold artefact ----------------
    print("\n== B: mixed spectrum (push + free chirps) ==")
    trajs_mixed = make_dataset(plant, n_push=30, n_free=8, seed=1, n_steps=110)
    Sm = spectrum_for(trajs_mixed, twin, space)
    st_m, fl_m = Sm.stiff_subspace(TOL), Sm.flat_subspace(TOL)
    print(f"  eff rank {Sm.effective_rank(TOL)}/18, null {Sm.null_dimension()}, "
          f"stiff {st_m.shape[1]}, flat {fl_m.shape[1]}")
    us, uf = unit_in(st_m), unit_in(fl_m)
    sweep = []
    for mag in (0.01, 0.03, 0.10, 0.45):
        ds = divergence(space.from_log(x0 + mag * us), cmd_eval)["whitened_rms"]
        df = divergence(space.from_log(x0 + mag * uf), cmd_eval)["whitened_rms"]
        sweep.append({"mag": mag, "stiff": ds, "flat": df, "ratio": ds / max(df, 1e-12)})
        print(f"  mag {mag:5.2f}  stiff {ds:7.3f}  flat {df:7.3f}  ratio {ds/max(df,1e-12):6.2f}x")

    # ---------------- renders ----------------
    print("\n== rendering ==")
    worlds = {
        "plant": make_plant(bus_cfg=CLEAN),
        "twin_stiff": World(theta=th_stiff, bus_cfg=CLEAN, space=space),
        "twin_flat": World(theta=th_flat, bus_cfg=CLEAN, space=space),
    }
    for cam in ("oblique", "overhead"):
        rr = mujoco.Renderer(worlds["plant"].model, height=H, width=W)
        worlds["plant"].reset(q_arm=cmd_eval[0])
        rr.update_scene(worlds["plant"].data, camera=cam)
        Image.fromarray(rr.render()).save(os.path.join(OUT, f"scene_{cam}.png"))
        print(f"  scene_{cam}.png")

    fp = render_frames(worlds["plant"], cmd_eval, "oblique")
    fs = render_frames(worlds["twin_stiff"], cmd_eval, "oblique")
    ff = render_frames(worlds["twin_flat"], cmd_eval, "oblique")
    combo = []
    for a, b, c in zip(fp, fs, ff):
        sheet = Image.new("RGB", (W * 3, H), (16, 18, 22))
        sheet.paste(label(a, "PLANT   ground truth"), (0, 0))
        sheet.paste(label(b, "TWIN   +57% in an IDENTIFIABLE direction"), (W, 0))
        sheet.paste(label(c, "TWIN   +57% in an UNIDENTIFIABLE direction"), (W * 2, 0))
        combo.append(sheet.resize((W * 3 * 2 // 3, H * 2 // 3), Image.LANCZOS))
    combo[0].save(os.path.join(OUT, "three_worlds.gif"), save_all=True,
                  append_images=combo[1:], duration=70, loop=0, optimize=True)
    sz = os.path.getsize(os.path.join(OUT, "three_worlds.gif")) / 1024
    print(f"  three_worlds.gif  ({len(combo)} frames, {sz:.0f} KB)")

    # ---------------- data export ----------------
    print("\n== exporting ==")
    runs = {}
    for tname, cmd in (("familiar", cmd_eval), ("novel", cmd_novel)):
        runs[tname] = {}
        for wname, w in worlds.items():
            r = run_logged(w, cmd)
            runs[tname][wname] = {
                "t": np.round(r["t"], 4).tolist(),
                "q": np.round(r["qtrue"], 5).tolist(),
                "box": np.round(r["box"], 5).tolist(),
                "tip": np.round(r["tip"], 5).tolist(),
            }

    plant_real = make_plant(bus_cfg=PLANT_BUS)
    br = run_logged(plant_real, cmd_eval)
    data = {
        "geometry": {
            "links": [0.18, 0.15, 0.10],
            "box_half": float(plant.model.geom("g_box").size[0]),
            "box_home": plant.model.body("box").pos[:2].tolist(),
            "tip_r": float(plant.model.geom("g_tip").size[0]),
            "pipe": {"xy": [0.355, -0.150], "r": 0.025},
            "bolt": {"xy": [0.170, -0.195], "r": 0.009},
            "target": {"xy": [0.400, 0.225], "r": 0.050},
            "reach": 0.43,
        },
        "param_names": list(space.names),
        "theta_nominal": space.nominal.tolist(),
        "spectra": {
            "push_only": {"eigvals": Sp.eigvals.tolist(),
                          "eigvecs": np.round(Sp.eigvecs, 4).tolist(),
                          "eff_rank": Sp.effective_rank(TOL), "null": Sp.null_dimension(),
                          "stiff_dim": int(st_p.shape[1]), "flat_dim": int(fl_p.shape[1]),
                          "n_push": 30, "n_free": 0},
            "mixed": {"eigvals": Sm.eigvals.tolist(),
                      "eigvecs": np.round(Sm.eigvecs, 4).tolist(),
                      "eff_rank": Sm.effective_rank(TOL), "null": Sm.null_dimension(),
                      "stiff_dim": int(st_m.shape[1]), "flat_dim": int(fl_m.shape[1]),
                      "n_push": 30, "n_free": 8},
        },
        "demoA": demoA,
        "sweep": sweep,
        "perturbations": {"magnitude_log": MAG, "pct": float(np.exp(MAG) - 1) * 100,
                          "d_stiff": np.round(d_stiff, 5).tolist(),
                          "d_flat": np.round(d_flat, 5).tolist()},
        "runs": runs,
        "bus_demo": {
            "t": np.round(br["t"], 4).tolist(),
            "cmd": np.round(br["cmd"][:, 1], 5).tolist(),
            "eff": np.round(br["eff"][:, 1], 5).tolist(),
            "q_obs": np.round(br["q"][:, 1], 6).tolist(),
            "q_true": np.round(br["qtrue"][:, 1], 6).tolist(),
            "stats": plant_real.bus.stats.summary(),
            "encoder_lsb": float(PLANT_BUS.encoder_resolution),
        },
        "bus_cfg": {"control_hz": PLANT_BUS.control_hz, "act_latency_s": PLANT_BUS.act_latency_s,
                    "act_jitter_s": PLANT_BUS.act_jitter_s, "dropout_p": PLANT_BUS.dropout_p,
                    "encoder_bits": PLANT_BUS.encoder_bits,
                    "sense_latency_s": PLANT_BUS.sense_latency_s},
        "noise_sigmas": dict(zip(CHANNELS, np.round(SIG, 8).tolist())),
    }
    p = os.path.join(OUT, "explainer_data.json")
    with open(p, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  explainer_data.json ({os.path.getsize(p)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
