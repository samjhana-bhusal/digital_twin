"""Setpoint plans and rollouts.

Plans are computed OFFLINE and then streamed, which is both faster and a better
model of a real servo bus: you push setpoints at a fixed rate and the servo does
what it can with them. Nothing here is allowed to read `World.true_state()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .plant import ARM_DOF, BOX_DOF, World


@dataclass
class Trajectory:
    t: np.ndarray                 # (T,)
    cmd: np.ndarray               # (T, 3)  commanded joint setpoints
    q_arm: np.ndarray             # (T, 3)  as observed (quantised, delayed)
    v_arm: np.ndarray             # (T, 3)
    box: np.ndarray               # (T, 3)  x, y, theta
    tip: np.ndarray               # (T, 2)
    true_qpos: np.ndarray         # (T, 6)  analysis only
    true_qvel: np.ndarray         # (T, 6)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.t.size)

    def observed(self) -> np.ndarray:
        """(T, 11): what an experimenter actually has. Identification uses this."""
        return np.concatenate([self.q_arm, self.v_arm, self.box, self.tip], axis=1)

    def box_displacement(self) -> float:
        return float(np.linalg.norm(self.box[-1, :2] - self.box[0, :2]))

    def made_contact(self) -> bool:
        return self.box_displacement() > 1e-4


# ------------------------------------------------------------------ plans

def plan_tip_line(
    world: World,
    p_start: np.ndarray,
    p_end: np.ndarray,
    n_steps: int,
    q_seed: np.ndarray | None = None,
) -> np.ndarray:
    """Straight line in tip space -> joint setpoints, via IK at each waypoint."""
    p_start = np.asarray(p_start, float)
    p_end = np.asarray(p_end, float)
    q = q_seed
    out = np.zeros((n_steps, ARM_DOF))
    for k in range(n_steps):
        s = k / max(n_steps - 1, 1)
        # smoothstep: zero velocity at both ends, so we excite the servo, not the planner
        s = s * s * (3 - 2 * s)
        target = (1 - s) * p_start + s * p_end
        q, _err = world.ik_tip(target, q_init=q)
        out[k] = q
    return out


def plan_chirp(
    q0: np.ndarray,
    n_steps: int,
    dt: float,
    amp: np.ndarray = np.array([0.25, 0.30, 0.35]),
    f0: float = 0.2,
    f1: float = 3.0,
) -> np.ndarray:
    """Logarithmic chirp per joint: the standard way to excite a plant across the
    band you care about. Used for free-space identification of arm parameters."""
    t = np.arange(n_steps) * dt
    T = max(t[-1], 1e-6)
    k = (f1 / f0) ** (1.0 / T)
    phase = 2 * np.pi * f0 * (k ** t - 1) / np.log(k)
    out = np.asarray(q0, float)[None, :] + amp[None, :] * np.sin(phase)[:, None]
    return out


def rollout(world: World, cmd: np.ndarray, q0: np.ndarray | None = None,
            box0: np.ndarray | None = None, meta: dict | None = None) -> Trajectory:
    """Stream `cmd` over the bus and log everything."""
    obs = world.reset(q_arm=q0 if q0 is not None else cmd[0], box=box0)
    T = cmd.shape[0]
    rec = {
        "t": np.zeros(T), "cmd": np.zeros((T, ARM_DOF)),
        "q_arm": np.zeros((T, ARM_DOF)), "v_arm": np.zeros((T, ARM_DOF)),
        "box": np.zeros((T, BOX_DOF)), "tip": np.zeros((T, 2)),
        "true_qpos": np.zeros((T, world.model.nq)), "true_qvel": np.zeros((T, world.model.nv)),
    }
    for k in range(T):
        rec["t"][k] = obs.t
        rec["cmd"][k] = cmd[k]
        rec["q_arm"][k] = obs.q_arm
        rec["v_arm"][k] = obs.v_arm
        rec["box"][k] = obs.box
        rec["tip"][k] = obs.tip
        ts = world.true_state()
        rec["true_qpos"][k] = ts[:world.model.nq]
        rec["true_qvel"][k] = ts[world.model.nq:]
        obs = world.step_control(cmd[k])

    md = {"bus": world.bus.stats.summary(), "control_hz": world.bus_cfg.control_hz}
    md.update(meta or {})
    return Trajectory(meta=md, **rec)


# ------------------------------------------------------------------ episode generators

def push_episode(
    world: World,
    rng: np.random.Generator,
    n_steps: int = 120,
    box0: np.ndarray | None = None,
) -> Trajectory:
    """Approach the box from a random angle and push through it.

    The push line is aimed to pass through the box centre with a random lateral
    offset, which is what produces the interesting cases: an off-centre push
    rotates the box, and box rotation is the signal that carries most of the
    information about contact friction.
    """
    box0 = np.zeros(BOX_DOF) if box0 is None else np.asarray(box0, float)
    # box nominal centre in world frame (body pos + slide offsets)
    centre = world.model.body("box").pos[:2] + box0[:2]
    half = float(world.model.geom("g_box").size[0])

    approach = rng.uniform(-np.pi, np.pi)
    u = np.array([np.cos(approach), np.sin(approach)])      # inward push direction
    perp = np.array([-u[1], u[0]])
    offset = rng.uniform(-0.8, 0.8) * half                   # lateral offset -> rotation
    standoff = half + 0.030
    travel = rng.uniform(0.045, 0.085)

    p_start = centre - u * standoff + perp * offset
    p_end = centre + u * travel + perp * offset

    cmd = plan_tip_line(world, p_start, p_end, n_steps)
    return rollout(world, cmd, box0=box0, meta={
        "kind": "push", "approach": float(approach), "offset": float(offset),
        "travel": float(travel), "p_start": p_start.tolist(), "p_end": p_end.tolist(),
    })


# Where the box is parked during free-space episodes. Far outside the arm's
# 0.43 m reach, so a chirp cannot clip it. Without this the "contact-free"
# episodes quietly include contact and stop isolating the arm parameters.
BOX_PARKED = np.array([0.0, -1.20, 0.0])


def free_episode(world: World, rng: np.random.Generator, n_steps: int = 200) -> Trajectory:
    """Free-space chirp with the box parked away. Isolates arm parameters from
    contact, which is what makes the arm block of the Hessian well conditioned."""
    q0 = np.array([rng.uniform(-0.3, 0.9), rng.uniform(-1.6, -0.6), rng.uniform(-0.4, 0.4)])
    cmd = plan_chirp(q0, n_steps, dt=world.bus_cfg.control_dt)
    cmd = np.clip(cmd, world.model.jnt_range[:ARM_DOF, 0], world.model.jnt_range[:ARM_DOF, 1])
    tr = rollout(world, cmd, box0=BOX_PARKED, meta={"kind": "free"})
    if tr.made_contact():  # invariant, not a hope
        raise RuntimeError("free episode made contact; BOX_PARKED is too close")
    return tr


def make_dataset(
    world: World,
    n_push: int = 40,
    n_free: int = 10,
    seed: int = 0,
    n_steps: int = 120,
) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    trajs = [push_episode(world, rng, n_steps=n_steps) for _ in range(n_push)]
    trajs += [free_episode(world, rng, n_steps=max(n_steps, 200)) for _ in range(n_free)]
    return trajs
