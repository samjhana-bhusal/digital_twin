"""Identifiability: the curvature of the fit, and what it says you cannot know.

R1's claim is that twin-to-plant transfer error is governed by the component of
parameter error lying in the FLAT directions of this Hessian, not by the total
parameter error. To test that you need the eigendecomposition, so that is what
this module produces.

The Hessian is taken in log-parameter space (see models/params.py), so its
eigenvalues are dimensionless and comparable across parameters of different
physical units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Spectrum:
    eigvals: np.ndarray      # ascending
    eigvecs: np.ndarray      # columns match eigvals
    names: tuple[str, ...]

    @property
    def condition_number(self) -> float:
        """Infinite when the data leaves an exact null direction, which is a
        statement about the model, not a numerical accident. See
        `stiff_condition_number` for the conditioning of what IS constrained."""
        if self.eigvals.size == 0 or self.eigvals[0] <= 0:
            return float("inf")
        return float(self.eigvals[-1] / self.eigvals[0])

    def stiff_condition_number(self, rel_tol: float = 1e-3) -> float:
        """Condition number restricted to the identifiable subspace."""
        keep = self.eigvals > rel_tol * self.eigvals[-1]
        ev = self.eigvals[keep]
        return float(ev[-1] / ev[0]) if ev.size else float("inf")

    def null_dimension(self) -> int:
        """Number of EXACTLY unconstrained directions (structural, not weak)."""
        return int(np.sum(self.eigvals <= 1e-12 * max(self.eigvals[-1], 1e-300)))

    def effective_rank(self, rel_tol: float = 1e-3) -> int:
        """How many parameter directions the data actually constrains."""
        return int(np.sum(self.eigvals > rel_tol * self.eigvals[-1]))

    def flat_subspace(self, rel_tol: float = 1e-3) -> np.ndarray:
        """Columns spanning the unidentifiable directions."""
        keep = self.eigvals <= rel_tol * self.eigvals[-1]
        return self.eigvecs[:, keep]

    def stiff_subspace(self, rel_tol: float = 1e-3) -> np.ndarray:
        keep = self.eigvals > rel_tol * self.eigvals[-1]
        return self.eigvecs[:, keep]

    def decompose_error(self, dlog: np.ndarray, rel_tol: float = 1e-3) -> dict[str, float]:
        """Split a log-parameter error into stiff and flat components.

        This is the measurement R1 reports: two twins with equal ||dlog|| but
        different stiff/flat splits should transfer very differently.
        """
        flat = self.flat_subspace(rel_tol)
        stiff = self.stiff_subspace(rel_tol)
        total = float(np.linalg.norm(dlog))
        f = float(np.linalg.norm(flat.T @ dlog)) if flat.size else 0.0
        s = float(np.linalg.norm(stiff.T @ dlog)) if stiff.size else 0.0
        return {
            "total": total,
            "stiff": s,
            "flat": f,
            "flat_fraction": f / total if total > 0 else 0.0,
        }

    def report(self, top: int = 4) -> str:
        lines = [
            f"effective rank {self.effective_rank()} / {self.eigvals.size}",
            f"condition number {self.condition_number:.3e}",
            "",
            "flattest directions (least identifiable):",
        ]
        for i in range(min(top, self.eigvals.size)):
            v = self.eigvecs[:, i]
            order = np.argsort(-np.abs(v))[:4]
            terms = "  ".join(f"{self.names[j]}{v[j]:+.2f}" for j in order)
            lines.append(f"  lam={self.eigvals[i]:.3e}   {terms}")
        return "\n".join(lines)


def gauss_newton_hessian(jac: np.ndarray) -> np.ndarray:
    """J^T J, the Gauss-Newton approximation. Cheap and positive semidefinite."""
    return jac.T @ jac


def finite_difference_jacobian(
    resid_fn, theta_log: np.ndarray, eps: float = 1e-4
) -> np.ndarray:
    """Central-difference Jacobian of the residual vector w.r.t. log parameters.

    Central differences, not forward: contact dynamics are only piecewise smooth
    and a one-sided stencil picks up a bias near contact-mode switches.
    """
    r0 = resid_fn(theta_log)
    J = np.zeros((r0.size, theta_log.size))
    for i in range(theta_log.size):
        dp = theta_log.copy(); dp[i] += eps
        dm = theta_log.copy(); dm[i] -= eps
        J[:, i] = (resid_fn(dp) - resid_fn(dm)) / (2 * eps)
    return J


def spectrum_from_jacobian(jac: np.ndarray, names: tuple[str, ...]) -> Spectrum:
    H = gauss_newton_hessian(jac)
    w, V = np.linalg.eigh(H)
    return Spectrum(eigvals=w, eigvecs=V, names=names)
