"""The identification objective.

Given data from the plant, how well does a candidate parameter vector reproduce
it? This is the function whose curvature R1 is about: its Hessian eigenspectrum
is the statement about what the data can and cannot pin down.

The loss is open-loop multi-step prediction error. The twin is handed the SAME
commanded setpoint sequence the plant received, reset to the plant's initial
state, and then left alone for the whole episode. That matters: a one-step
(teacher-forced) loss is far better conditioned than reality, so fitting it
produces a twin that looks identified and still diverges over a rollout.

Residuals are divided by the OBSERVATION NOISE of each channel, not by the
spread of the data. Two reasons. First, it makes the loss a proper negative log
likelihood, so its Hessian is Fisher information and R1's eigenvalues mean
something statistical rather than something arbitrary. Second, data-driven
whitening quietly fails here: the box has not moved yet for most of a pushing
episode, so a robust spread estimate of box position collapses toward zero and
the loss ends up dominated by whichever channel happened to be stillest.
"""

from __future__ import annotations

import numpy as np

from ..bus import BusConfig
from ..models.params import ParamSpace
from ..plant import World, make_twin
from ..rollout import Trajectory, rollout


def fitting_twin(theta: np.ndarray | None = None, data_bus: BusConfig | None = None) -> World:
    """A twin configured for identification: same interface as the plant that
    produced the data, but with the randomness and quantisation stripped out.
    See BusConfig.deterministic for why."""
    cfg = (data_bus or BusConfig()).deterministic()
    return make_twin(theta=theta, bus_cfg=cfg)


# Observation channels, in the order Trajectory.observed() emits them.
CHANNELS = ("q1", "q2", "q3", "v1", "v2", "v3", "box_x", "box_y", "box_t", "tip_x", "tip_y")

# Sensor assumptions, stated once and reused. These are the numbers a real rig
# would quote from its datasheets, and every one is an assumption the project
# should be able to point at.
VISION_POS_SIGMA_M = 0.5e-3          # pose estimate of the box from overhead view
VISION_ANG_SIGMA_RAD = np.radians(0.5)
ARM_REACH_M = 0.43                   # 0.18 + 0.15 + 0.10, for propagating q noise to the tip


def noise_sigmas(bus_cfg: BusConfig) -> np.ndarray:
    """Per-channel observation noise sigma, derived from the interface itself.

    Joint angles are limited by encoder quantisation, whose error is uniform on
    one LSB, hence sigma = LSB / sqrt(12). Velocities are differenced encoder
    reads, so their noise is amplified by sqrt(2)/dt. Tip position is forward
    kinematics of noisy angles, so it inherits roughly reach * sigma_q. Box pose
    comes from vision and is quoted independently.
    """
    lsb = bus_cfg.encoder_resolution
    if lsb <= 0:                      # perfect/deterministic bus: assume float32-ish
        sigma_q = 1e-6
    else:
        sigma_q = lsb / np.sqrt(12.0)
    sigma_v = sigma_q * np.sqrt(2.0) / bus_cfg.control_dt
    sigma_tip = ARM_REACH_M * sigma_q
    return np.array([
        sigma_q, sigma_q, sigma_q,
        sigma_v, sigma_v, sigma_v,
        VISION_POS_SIGMA_M, VISION_POS_SIGMA_M, VISION_ANG_SIGMA_RAD,
        sigma_tip, sigma_tip,
    ])


def channel_scales(trajs: list[Trajectory]) -> np.ndarray:
    """Data-driven fallback. Kept for comparison only -- `noise_sigmas` is the
    scaling the project actually uses; see the module docstring for why."""
    obs = np.concatenate([tr.observed() for tr in trajs], axis=0)
    mad = np.median(np.abs(obs - np.median(obs, axis=0)), axis=0)
    return np.maximum(1.4826 * mad, 1e-4)


def predict(world: World, tr: Trajectory) -> np.ndarray:
    """Replay an episode's commands in `world` from the episode's initial state."""
    q0 = tr.true_qpos[0, :3]
    box0 = tr.true_qpos[0, 3:6]
    sim = rollout(world, tr.cmd, q0=q0, box0=box0)
    return sim.observed()


def residuals(
    theta: np.ndarray,
    trajs: list[Trajectory],
    world: World,
    scales: np.ndarray,
) -> np.ndarray:
    """Flat whitened residual vector; `scipy.optimize.least_squares` eats this."""
    world.set_theta(theta)
    out = []
    for tr in trajs:
        pred = predict(world, tr)
        out.append(((pred - tr.observed()) / scales[None, :]).ravel())
    return np.concatenate(out)


def nll(theta_log: np.ndarray, trajs: list[Trajectory], world: World,
        scales: np.ndarray, space: ParamSpace) -> float:
    """Scalar objective in LOG parameter space. Half-sum-of-squares, so its
    Hessian is the Gauss-Newton/Fisher information up to noise scale."""
    theta = space.from_log(theta_log)
    r = residuals(theta, trajs, world, scales)
    return 0.5 * float(r @ r) / r.size
