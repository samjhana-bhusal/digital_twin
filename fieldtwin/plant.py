"""`World`: one MuJoCo world, reachable only through a bus.

The plant and the twin are both `World` objects. They differ in their parameter
vector, and optionally in whether they model the bus at all -- which is the
point of R-bus: a twin that assumes a perfect interface is a *different kind* of
wrong from a twin with mis-identified friction, and the project measures both.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .bus import BusConfig, ServoBus
from .models.params import SCENE_XML, ParamSpace

ARM_DOF = 3          # j1, j2, j3
BOX_DOF = 3          # bx, by, bt


@dataclass
class Observation:
    t: float
    q_arm: np.ndarray        # quantised, delayed joint angles (3,)
    v_arm: np.ndarray
    box: np.ndarray          # box (x, y, theta) -- a vision system supplies this
    tip: np.ndarray          # fingertip (x, y) from forward kinematics

    def state(self) -> np.ndarray:
        """The 12-dim vector the world model in R2 predicts."""
        return np.concatenate([self.q_arm, self.v_arm, self.box, self.tip])


class World:
    def __init__(
        self,
        theta: np.ndarray | None = None,
        bus_cfg: BusConfig | None = None,
        xml_path: str = SCENE_XML,
        space: ParamSpace | None = None,
    ):
        self.space = space or ParamSpace.from_xml(xml_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.theta = self.space.nominal.copy() if theta is None else np.asarray(theta, float).copy()
        self.space.set(self.model, self.theta)
        self.data = mujoco.MjData(self.model)
        if hasattr(mujoco, "mj_setConst"):
            mujoco.mj_setConst(self.model, self.data)

        self.bus_cfg = bus_cfg if bus_cfg is not None else BusConfig()
        self.bus = ServoBus(self.bus_cfg, nu=self.model.nu, nq=self.model.nq)
        self._tip_site = self.model.site("s_tip").id
        self._sim_dt = float(self.model.opt.timestep)
        self.reset()

    def set_theta(self, theta: np.ndarray) -> None:
        """Rebind parameters. Must go through here, not through space.set alone:
        changing a mass invalidates MuJoCo's derived constants (subtree masses,
        inertia-weighted terms), and mj_setConst is what recomputes them."""
        self.theta = np.asarray(theta, float).copy()
        self.space.set(self.model, self.theta)
        if hasattr(mujoco, "mj_setConst"):
            mujoco.mj_setConst(self.model, self.data)

    # ---------------- lifecycle ----------------

    def reset(self, q_arm: np.ndarray | None = None, box: np.ndarray | None = None) -> Observation:
        mujoco.mj_resetData(self.model, self.data)
        if q_arm is not None:
            self.data.qpos[:ARM_DOF] = q_arm
        if box is not None:
            self.data.qpos[ARM_DOF:ARM_DOF + BOX_DOF] = box
        mujoco.mj_forward(self.model, self.data)
        self.bus.reset(self.data.qpos[:ARM_DOF].copy())
        self.data.ctrl[:] = self.data.qpos[:ARM_DOF]
        self.bus.record(self.data.time, self.data.qpos, self.data.qvel)
        return self.observe()

    # ---------------- stepping ----------------

    def step_control(self, ctrl: np.ndarray) -> Observation:
        """Issue one command and advance exactly one control period."""
        t0 = self.data.time
        self.bus.send(t0, ctrl)
        n_sub = max(1, int(round(self.bus_cfg.control_dt / self._sim_dt)))
        for _ in range(n_sub):
            self.data.ctrl[:] = self.bus.effective_ctrl(self.data.time)
            mujoco.mj_step(self.model, self.data)
            self.bus.record(self.data.time, self.data.qpos, self.data.qvel)
        return self.observe()

    # ---------------- sensing ----------------

    def observe(self) -> Observation:
        q, v = self.bus.observe(self.data.time)
        return Observation(
            t=float(self.data.time),
            q_arm=q[:ARM_DOF].copy(),
            v_arm=v[:ARM_DOF].copy(),
            box=self.data.qpos[ARM_DOF:ARM_DOF + BOX_DOF].copy(),
            tip=self.tip_xy(),
        )

    def true_state(self) -> np.ndarray:
        """Unquantised, undelayed ground truth. For analysis only -- a controller
        that reads this is cheating, and the harness never passes it to one."""
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def tip_xy(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_site][:2].copy()

    # ---------------- task space ----------------

    def tip_jacobian(self) -> np.ndarray:
        """2x3 planar Jacobian of the fingertip w.r.t. the arm joints."""
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._tip_site)
        return jacp[:2, :ARM_DOF].copy()

    def ik_tip(
        self,
        target_xy: np.ndarray,
        q_init: np.ndarray | None = None,
        iters: int = 200,
        step: float = 0.5,
        tol: float = 1e-5,
    ) -> tuple[np.ndarray, float]:
        """Damped-least-squares IK on a scratch copy. Returns (q_arm, residual)."""
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        if q_init is not None:
            d.qpos[:ARM_DOF] = q_init
        target = np.asarray(target_xy, float)
        jacp = np.zeros((3, self.model.nv))
        err = np.inf
        for _ in range(iters):
            mujoco.mj_forward(self.model, d)
            cur = d.site_xpos[self._tip_site][:2]
            e = target - cur
            err = float(np.linalg.norm(e))
            if err < tol:
                break
            mujoco.mj_jacSite(self.model, d, jacp, None, self._tip_site)
            J = jacp[:2, :ARM_DOF]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(2), e)
            d.qpos[:ARM_DOF] = np.clip(
                d.qpos[:ARM_DOF] + step * dq,
                self.model.jnt_range[:ARM_DOF, 0],
                self.model.jnt_range[:ARM_DOF, 1],
            )
        return d.qpos[:ARM_DOF].copy(), err


def make_plant(rel_sigma: float = 0.0, seed: int = 0, bus_cfg: BusConfig | None = None) -> World:
    """The ground-truth world. Its parameters are what the twin must recover."""
    from .models.params import perturb

    space = ParamSpace.from_xml()
    theta = space.nominal.copy()
    if rel_sigma > 0:
        theta = perturb(space, theta, rel_sigma, np.random.default_rng(seed))
    return World(theta=theta, bus_cfg=bus_cfg, space=space)


def make_twin(
    theta: np.ndarray | None = None,
    models_bus: bool = True,
    bus_cfg: BusConfig | None = None,
) -> World:
    """The twin. `models_bus=False` is the common real-world mistake: a twin that
    silently assumes commands apply instantly and encoders are exact."""
    space = ParamSpace.from_xml()
    cfg = (bus_cfg or BusConfig()) if models_bus else BusConfig.perfect()
    return World(theta=theta, bus_cfg=cfg, space=space)
