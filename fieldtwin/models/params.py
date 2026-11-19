"""The parameter space that separates the plant from the twin.

Everything in `PARAM_SPECS` is a quantity a real experimenter would NOT know and
would have to identify from data. Everything outside it (link lengths, geometry,
timestep, the kinematic tree) is treated as known, because on real hardware you
measure those with a ruler and a datasheet.

Two deliberate choices:

1.  Parameters are handled in LOG space. All of them are physically positive, so
    log space keeps the optimiser inside the feasible set for free and -- more
    importantly for R1 -- makes the Hessian scale-free. A Hessian in raw units
    mixes kg, N*m*s/rad and dimensionless friction, so its eigenvalues would be
    an artefact of unit choice rather than a statement about identifiability.

2.  Changing a body's mass rescales its inertia tensor by the same factor. The
    geometry is fixed, so mass and inertia are not independent; letting them
    drift apart would silently invent bodies with impossible density.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

SCENE_XML = os.path.join(os.path.dirname(__file__), "scene.xml")

# (name, kind, mjcf element name)
PARAM_SPECS: tuple[tuple[str, str, str], ...] = (
    # --- arm inertial ---
    ("m_link1", "body_mass", "link1"),
    ("m_link2", "body_mass", "link2"),
    ("m_link3", "body_mass", "link3"),
    # --- arm dissipation ---
    ("d_j1", "dof_damping", "j1"),
    ("d_j2", "dof_damping", "j2"),
    ("d_j3", "dof_damping", "j3"),
    ("f_j1", "dof_frictionloss", "j1"),
    ("f_j2", "dof_frictionloss", "j2"),
    ("f_j3", "dof_frictionloss", "j3"),
    # --- servo gains ---
    ("kp_a1", "actuator_kp", "a1"),
    ("kp_a2", "actuator_kp", "a2"),
    ("kp_a3", "actuator_kp", "a3"),
    # --- object and table ---
    ("m_box", "body_mass", "box"),
    ("f_bx", "dof_frictionloss", "bx"),
    ("f_by", "dof_frictionloss", "by"),
    ("f_bt", "dof_frictionloss", "bt"),
    # --- contact ---
    ("mu_box", "geom_friction_t", "g_box"),
    ("mu_tip", "geom_friction_t", "g_tip"),
)

PARAM_NAMES: tuple[str, ...] = tuple(s[0] for s in PARAM_SPECS)
N_PARAMS = len(PARAM_SPECS)


def _ids(model: mujoco.MjModel) -> list[int]:
    out = []
    for _name, kind, elem in PARAM_SPECS:
        if kind == "body_mass":
            out.append(model.body(elem).id)
        elif kind in ("dof_damping", "dof_frictionloss"):
            out.append(int(model.joint(elem).dofadr[0]))
        elif kind == "actuator_kp":
            out.append(model.actuator(elem).id)
        elif kind == "geom_friction_t":
            out.append(model.geom(elem).id)
        else:  # pragma: no cover
            raise ValueError(f"unknown parameter kind {kind!r}")
    return out


@dataclass
class ParamSpace:
    """Maps a flat (log-)parameter vector onto an MjModel, and back."""

    names: tuple[str, ...]
    nominal: np.ndarray            # parameter values as written in scene.xml
    _ids: list[int]
    _ref_mass: np.ndarray          # pristine body_mass, for inertia rescaling
    _ref_inertia: np.ndarray       # pristine body_inertia

    @classmethod
    def from_xml(cls, xml_path: str = SCENE_XML) -> "ParamSpace":
        m = mujoco.MjModel.from_xml_path(xml_path)
        ids = _ids(m)
        space = cls(
            names=PARAM_NAMES,
            nominal=np.zeros(N_PARAMS),
            _ids=ids,
            _ref_mass=m.body_mass.copy(),
            _ref_inertia=m.body_inertia.copy(),
        )
        space.nominal = space.get(m)
        return space

    # ---------------- read / write ----------------

    def get(self, model: mujoco.MjModel) -> np.ndarray:
        theta = np.zeros(N_PARAMS)
        for i, (_name, kind, _elem) in enumerate(PARAM_SPECS):
            j = self._ids[i]
            if kind == "body_mass":
                theta[i] = model.body_mass[j]
            elif kind == "dof_damping":
                theta[i] = model.dof_damping[j]
            elif kind == "dof_frictionloss":
                theta[i] = model.dof_frictionloss[j]
            elif kind == "actuator_kp":
                theta[i] = model.actuator_gainprm[j, 0]
            elif kind == "geom_friction_t":
                theta[i] = model.geom_friction[j, 0]
        return theta

    def set(self, model: mujoco.MjModel, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (N_PARAMS,):
            raise ValueError(f"expected {N_PARAMS} parameters, got {theta.shape}")
        if np.any(theta <= 0):
            raise ValueError("all fieldtwin parameters are strictly positive")

        for i, (_name, kind, _elem) in enumerate(PARAM_SPECS):
            j = self._ids[i]
            v = float(theta[i])
            if kind == "body_mass":
                model.body_mass[j] = v
                # geometry is fixed, so inertia must track mass exactly
                scale = v / self._ref_mass[j]
                model.body_inertia[j] = self._ref_inertia[j] * scale
            elif kind == "dof_damping":
                model.dof_damping[j] = v
            elif kind == "dof_frictionloss":
                model.dof_frictionloss[j] = v
            elif kind == "actuator_kp":
                # MuJoCo position actuator: gain = kp, bias = -kp on position
                model.actuator_gainprm[j, 0] = v
                model.actuator_biasprm[j, 1] = -v
            elif kind == "geom_friction_t":
                model.geom_friction[j, 0] = v

    # ---------------- log space ----------------

    def to_log(self, theta: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(theta, dtype=float))

    def from_log(self, x: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(x, dtype=float))

    def index(self, name: str) -> int:
        return self.names.index(name)

    def describe(self, theta: np.ndarray) -> str:
        w = max(len(n) for n in self.names)
        return "\n".join(
            f"  {n:<{w}}  {v:>10.5g}   (nominal {nv:>10.5g},  x{v / nv:.3f})"
            for n, v, nv in zip(self.names, theta, self.nominal)
        )


def perturb(
    space: ParamSpace,
    theta0: np.ndarray,
    rel_sigma: float = 0.25,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Log-normal multiplicative perturbation: the honest model of 'our twin is
    wrong by roughly 25% per parameter, and we do not know in which direction'."""
    rng = rng or np.random.default_rng(0)
    return space.from_log(space.to_log(theta0) + rel_sigma * rng.standard_normal(theta0.size))
