"""Trajectory persistence with enough provenance to re-run a result.

Every saved dataset carries the parameter vector that generated it, the bus
configuration, the MuJoCo version and a hash of the scene XML. A result that
cannot name the scene it came from is not a result.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

import mujoco
import numpy as np

from .bus import BusConfig
from .models.params import PARAM_NAMES, SCENE_XML
from .rollout import Trajectory

ARRAY_KEYS = ("t", "cmd", "q_arm", "v_arm", "box", "tip", "true_qpos", "true_qvel")


def scene_hash(xml_path: str = SCENE_XML) -> str:
    with open(xml_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def save_dataset(
    path: str,
    trajs: list[Trajectory],
    theta: np.ndarray,
    bus_cfg: BusConfig,
    note: str = "",
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat: dict[str, np.ndarray] = {}
    for i, tr in enumerate(trajs):
        for k in ARRAY_KEYS:
            flat[f"{i}/{k}"] = getattr(tr, k)
    flat["theta"] = np.asarray(theta, float)
    np.savez_compressed(path, **flat)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_traj": len(trajs),
        "note": note,
        "scene_sha256_16": scene_hash(),
        "mujoco_version": mujoco.__version__,
        "param_names": list(PARAM_NAMES),
        "theta": np.asarray(theta, float).tolist(),
        "bus_cfg": asdict(bus_cfg),
        "episodes": [tr.meta for tr in trajs],
    }
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def load_dataset(path: str) -> tuple[list[Trajectory], dict]:
    z = np.load(path)
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path) as f:
        meta = json.load(f)
    trajs = []
    for i in range(meta["n_traj"]):
        kw = {k: z[f"{i}/{k}"] for k in ARRAY_KEYS}
        trajs.append(Trajectory(meta=meta["episodes"][i], **kw))
    return trajs, meta
