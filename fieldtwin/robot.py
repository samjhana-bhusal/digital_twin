"""Arm control: inverse kinematics, Cartesian moves, gripper.

IK is `mink` (differential IK over MuJoCo models) rather than hand-rolled. What
is written here is the layer above it — the pose conventions, the move
primitives, and the convergence and reachability checks a skill can rely on.
"""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from . import arena

ARM_DOF = 7
GRIPPER_ACT = 7          # actuator index of the Franka hand
GRIPPER_OPEN = 255.0     # ctrl units; the hand tendon is driven 0..255
GRIPPER_CLOSED = 0.0

# Base-joint angle that swings the arm behind the robot, clear of the overhead
# camera's frustum. The arm is the largest occluder in its own sensor image.
LOOK_YAW = -2.2


def top_down(yaw: float) -> np.ndarray:
    """Rotation for a top-down grasp with the fingers opening along `yaw`.

    The Franka hand approaches along its own +z and the fingers translate along
    its ±y. Rotating pi about x points the approach axis at the table; the yaw
    then spins the fingers to straddle the object.
    """
    Rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return Rz @ Rx


@dataclass
class MoveResult:
    reached: bool
    pos_err: float
    rot_err: float
    steps: int


class Panda:
    """Position-controlled Panda with a parallel gripper."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 control_hz: float = 50.0):
        self.model, self.data = model, data
        self.control_dt = 1.0 / control_hz
        self._substeps = max(1, int(round(self.control_dt / model.opt.timestep)))

        self.ik_model = arena.arm_model()
        self.config = mink.Configuration(self.ik_model)
        self.tcp_task = mink.FrameTask(
            frame_name="tcp", frame_type="site",
            position_cost=1.0, orientation_cost=0.4, lm_damping=1.0)
        # Low cost on purpose: the posture task should only break ties between
        # equally good solutions (keeping the elbow sane), never pull the TCP
        # off target. At cost 2e-2 it produced a standing ~2 mm position error.
        self.posture_task = mink.PostureTask(self.ik_model, cost=5e-4)
        self.posture_task.set_target(self.ik_model.key_qpos[0].copy())
        self.limits = [mink.ConfigurationLimit(self.ik_model)]
        self._tcp_site = model.site("tcp").id
        # Optional per-control-step hook, used for rendering and logging. Kept
        # out of the skill layer so skills never know whether they are observed.
        self.on_step = None

    # ---------------- state ----------------

    @property
    def q_arm(self) -> np.ndarray:
        return self.data.qpos[:ARM_DOF].copy()

    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp_site].copy()

    def tcp_rot(self) -> np.ndarray:
        return self.data.site_xmat[self._tcp_site].reshape(3, 3).copy()

    def home(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:ARM_DOF] = self.model.key_qpos[0][:ARM_DOF]
        self.data.ctrl[GRIPPER_ACT] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)

    # ---------------- inverse kinematics ----------------

    def solve(self, pos: np.ndarray, R: np.ndarray,
              q_seed: np.ndarray | None = None,
              iters: int = 400, tol_pos: float = 1.0e-3,
              tol_rot: float = 1e-2) -> tuple[np.ndarray, bool, float, float]:
        """Joint angles putting the TCP at (pos, R). Returns (q, ok, e_pos, e_rot)."""
        q0 = self.ik_model.key_qpos[0].copy()
        if q_seed is not None:
            q0[:ARM_DOF] = q_seed
        self.config.update(q0)
        T = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(np.asarray(R, float)), np.asarray(pos, float))
        self.tcp_task.set_target(T)
        dt = 0.02
        prev = np.inf
        for _ in range(iters):
            v = mink.solve_ik(self.config, [self.tcp_task, self.posture_task],
                              dt, solver="daqp", damping=1e-3, limits=self.limits)
            self.config.integrate_inplace(v, dt)
            err = self.tcp_task.compute_error(self.config)
            e_pos = float(np.linalg.norm(err[:3]))
            e_rot = float(np.linalg.norm(err[3:]))
            if e_pos < tol_pos and e_rot < tol_rot:
                return self.config.q[:ARM_DOF].copy(), True, e_pos, e_rot
            if abs(prev - e_pos) < 1e-9 and e_pos > tol_pos:
                break                      # plateaued short of the target
            prev = e_pos
        return self.config.q[:ARM_DOF].copy(), False, e_pos, e_rot

    # ---------------- motion ----------------

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            for _ in range(self._substeps):
                mujoco.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step(self)

    def goto_q(self, q_target: np.ndarray, steps: int = 60,
               settle: int = 12) -> None:
        """Ramp the position setpoints from current to target, then settle.

        A ramp rather than a step: commanding a distant setpoint in one go makes
        the position actuators yank, which throws objects off the table on
        contact and makes every result depend on actuator gains instead of on
        the plan.
        """
        q_start = self.data.ctrl[:ARM_DOF].copy()
        q_target = np.asarray(q_target, float)
        for k in range(1, steps + 1):
            a = k / steps
            a = a * a * (3 - 2 * a)            # smoothstep
            self.data.ctrl[:ARM_DOF] = (1 - a) * q_start + a * q_target
            self.step()
        for _ in range(settle):
            self.step()

    def move_to(self, pos, R, steps: int = 60, settle: int = 12) -> MoveResult:
        q, ok, e_pos, e_rot = self.solve(pos, R, q_seed=self.q_arm)
        if not ok:
            return MoveResult(False, e_pos, e_rot, 0)
        self.goto_q(q, steps=steps, settle=settle)
        achieved = float(np.linalg.norm(self.tcp_pos() - np.asarray(pos, float)))
        return MoveResult(True, achieved, e_rot, steps)

    # ---------------- gripper ----------------

    def set_gripper(self, value: float, steps: int = 24) -> None:
        self.data.ctrl[GRIPPER_ACT] = float(value)
        self.step(steps)

    def open_gripper(self, steps: int = 24) -> None:
        self.set_gripper(GRIPPER_OPEN, steps)

    def close_gripper(self, steps: int = 36) -> None:
        self.set_gripper(GRIPPER_CLOSED, steps)

    def finger_gap(self) -> float:
        """Distance between the fingertips, in metres."""
        return float(self.data.qpos[ARM_DOF] + self.data.qpos[ARM_DOF + 1])

    # ---------------- getting out of its own way ----------------

    def park_for_view(self, clear_z: float = 0.42, steps: int = 60) -> None:
        """Lift, then swing the whole arm out of the overhead camera's frame.

        A tabletop arm stands between its own ceiling camera and the table, so
        "look at the scene" is not a free operation -- it is a motion, and one
        that has to be safe. Lifting before swinging is the part that matters:
        rotating the base joint from a low pose sweeps the forearm through the
        workspace at object height and scatters the scene, which would show up
        as collateral damage caused by *perceiving* rather than by acting.

        Note this cannot be `home()`: `mj_resetDataKeyframe` rewrites the whole
        of `qpos`, which would teleport every cube back to where it spawned.
        """
        here = self.tcp_pos()
        self.move_to(np.array([here[0], here[1], max(here[2], clear_z)]),
                     self.tcp_rot(), steps=max(steps // 2, 20))
        q = self.model.key_qpos[0][:ARM_DOF].copy()
        q[0] = LOOK_YAW
        self.goto_q(q, steps=steps)

    def ready(self, steps: int = 70) -> None:
        """Return to the keyframe arm pose without resetting anything else.

        The counterpart to `park_for_view`. Reaching for an object from the
        parked pose means starting every approach from a configuration swung
        out behind the robot, which is a poor IK seed and a long traverse: it
        cost 24 of 150 episodes to `object_slipped` before this existed. Looking
        should not make the arm worse at its job, so looking ends where it
        began.

        `home()` would do this too and would also rewrite `qpos`, teleporting
        every cube back to where it spawned.
        """
        self.goto_q(self.model.key_qpos[0][:ARM_DOF].copy(), steps=steps)
