"""A simulated servo bus: the software half of hardware integration.

On real hardware you never touch the plant. You touch a bus, and the bus is
late, quantised, rate-limited and occasionally forgetful. Every one of those
effects shows up in logged data as *apparent dynamics*, which is why a naive
identification run will happily absorb your latency into a damping coefficient
and then report a beautifully converged fit.

This module makes those effects explicit and separately switchable, so the
project can answer: of the measured plant/twin divergence, how much is physics
we got wrong and how much is interface we failed to model?

Modelled, all independently toggleable:
  * finite control rate (commands issued every 1/control_hz, held in between)
  * actuation latency with jitter (command applies some time after it is sent)
  * command dropout (lost packet -> servo holds its previous setpoint)
  * encoder quantisation (position read back on a finite bit grid)
  * sensing latency (the state you read is the state from some time ago)

`BusConfig.perfect()` disables all of it, which is the control condition.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class BusConfig:
    control_hz: float = 50.0
    act_latency_s: float = 0.012
    act_jitter_s: float = 0.004
    dropout_p: float = 0.01
    encoder_bits: int = 12
    encoder_span_rad: float = 2 * np.pi   # full-turn encoder, as on a hobby servo
    sense_latency_s: float = 0.004
    seed: int = 0

    @staticmethod
    def perfect() -> "BusConfig":
        """No interface error at all. The reference condition for R-bus."""
        return BusConfig(
            control_hz=500.0,
            act_latency_s=0.0,
            act_jitter_s=0.0,
            dropout_p=0.0,
            encoder_bits=0,          # 0 => no quantisation
            sense_latency_s=0.0,
        )

    def deterministic(self) -> "BusConfig":
        """Same interface, no randomness and no quantisation.

        This is the configuration a TWIN should use when it is being fitted.
        Jitter and dropout would make the loss stochastic, so a finite-difference
        Jacobian would measure the random seed rather than the parameters. And
        encoder quantisation is a step function: a small parameter perturbation
        either does not move the quantised output at all (zero gradient) or moves
        it by a whole LSB (huge gradient). Both effects belong on the measured
        DATA, as observation noise, not inside the predictor.

        Mean latency, control rate and sensing delay are kept, because those are
        deterministic and genuinely change the dynamics the data reflects.
        """
        return replace(self, act_jitter_s=0.0, dropout_p=0.0, encoder_bits=0)

    def without_latency(self) -> "BusConfig":
        return replace(self, act_latency_s=0.0, act_jitter_s=0.0, sense_latency_s=0.0)

    def without_quantisation(self) -> "BusConfig":
        return replace(self, encoder_bits=0)

    def without_dropout(self) -> "BusConfig":
        return replace(self, dropout_p=0.0)

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def encoder_resolution(self) -> float:
        if self.encoder_bits <= 0:
            return 0.0
        return self.encoder_span_rad / (2 ** self.encoder_bits)


@dataclass
class BusStats:
    sent: int = 0
    dropped: int = 0
    applied: int = 0
    latencies: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        lat = np.asarray(self.latencies) if self.latencies else np.zeros(1)
        return {
            "sent": float(self.sent),
            "dropped": float(self.dropped),
            "drop_rate": float(self.dropped / max(self.sent, 1)),
            "latency_mean_s": float(lat.mean()),
            "latency_p95_s": float(np.quantile(lat, 0.95)),
            "latency_max_s": float(lat.max()),
        }


class ServoBus:
    """Sits between a controller and an MjData. Not thread-safe, by design."""

    def __init__(self, cfg: BusConfig, nu: int, nq: int):
        self.cfg = cfg
        self.nu = nu
        self.nq = nq
        self._rng = np.random.default_rng(cfg.seed)
        self._pending: list[tuple[float, int, np.ndarray]] = []
        self._seq = 0
        self._held = np.zeros(nu)             # servo's current setpoint
        self._history: list[tuple[float, np.ndarray, np.ndarray]] = []
        self.stats = BusStats()

    def reset(self, ctrl0: np.ndarray | None = None) -> None:
        self._pending.clear()
        self._seq = 0
        self._held = np.zeros(self.nu) if ctrl0 is None else np.asarray(ctrl0, float).copy()
        self._history.clear()
        self.stats = BusStats()

    # ---------------- controller -> plant ----------------

    def send(self, t: float, ctrl: np.ndarray) -> None:
        """Issue a command at time t. It arrives later, or never."""
        self.stats.sent += 1
        if self._rng.random() < self.cfg.dropout_p:
            self.stats.dropped += 1
            return
        lat = self.cfg.act_latency_s
        if self.cfg.act_jitter_s > 0:
            lat = max(0.0, lat + self.cfg.act_jitter_s * self._rng.standard_normal())
        self.stats.latencies.append(lat)
        self._seq += 1
        heapq.heappush(self._pending, (t + lat, self._seq, np.asarray(ctrl, float).copy()))

    def effective_ctrl(self, t: float) -> np.ndarray:
        """The setpoint the servo is actually holding at time t."""
        while self._pending and self._pending[0][0] <= t:
            _, _, ctrl = heapq.heappop(self._pending)
            self._held = ctrl
            self.stats.applied += 1
        return self._held

    # ---------------- plant -> controller ----------------

    def record(self, t: float, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self._history.append((t, np.asarray(qpos, float).copy(), np.asarray(qvel, float).copy()))

    def observe(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Quantised, delayed readback. What the controller is allowed to see."""
        if not self._history:
            return np.zeros(self.nq), np.zeros(self.nq)
        t_want = t - self.cfg.sense_latency_s
        # history is time-ordered; walk back to the newest sample at/before t_want
        idx = len(self._history) - 1
        while idx > 0 and self._history[idx][0] > t_want:
            idx -= 1
        _, qpos, qvel = self._history[idx]
        res = self.cfg.encoder_resolution
        if res > 0:
            qpos = np.round(qpos / res) * res
        return qpos, qvel
