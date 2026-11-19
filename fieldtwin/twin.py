"""The digital twin, and the loop that uses it.

**Why the twin must be wrong.** The obvious way to build this is to copy the
plant's state into a second `MjData` and roll the plan out. Do that and the
experiment is worthless: MuJoCo is deterministic, so an identical model handed
an identical state and an identical control sequence reproduces the plant
exactly, and "did the twin predict the outcome?" answers itself. The 2x2 would
be diagonal by construction, in the same way the first S2 benchmark scored
150/150 and measured nothing.

A twin is a *model*, and the two things that make real models wrong are both
represented here:

  * **It does not know the state.** The twin is initialised from the overhead
    camera (`perceive.py`), not from `qpos`. Positions come in a couple of
    millimetres off, and cube *orientation* does not come in at all -- a
    top-down blob says nothing about yaw -- so the twin plans its grasp against
    a cube it believes is square to the table when it is not.
  * **It does not know the parameters.** `TwinFidelity` holds a deliberate
    mis-specification of mass, friction and joint damping, as a signed magnitude
    along a direction in log-parameter space, so that a sweep through zero and
    out the other side is one experiment rather than two.

Neither source can be switched off entirely, and the names here say so:
`exact_physics()` gives the twin true parameters but leaves its object positions
coming from the camera, and `known_yaw` hands it the one piece of state the
camera cannot supply. Crossing those two is S5's factorial, and it found them to
be substitutes rather than independent -- see docs/findings.md F9.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import plan as planning
from . import tasks
from .arena import SceneSpec, build
from .perceive import StateView
from .robot import ARM_DOF, Panda


# The mis-specification lives in log space, so that "twice as wrong" means the
# same thing for a multiplier above 1 and one below it. A fidelity is then a
# direction and a signed magnitude along it, which is what makes a sweep
# possible: the S4 default sits at +0.22 along `nominal`, and -0.22 is its exact
# mirror -- a twin wrong by the same amount in the opposite sense.
PARAMS = ("cube_mass", "cube_friction", "joint_damping")

#: Named directions in log-parameter space. `nominal` is the S4 default's
#: direction: heavier, more slippery, more damped -- a twin that expects objects
#: to be harder to hold than they are.
_RAW_DIRECTIONS = {
    "nominal":  (np.log(1.12), np.log(0.85), np.log(1.10)),
    "mass":     (1.0, 0.0, 0.0),
    "friction": (0.0, 1.0, 0.0),
    "damping":  (0.0, 0.0, 1.0),
}
DIRECTIONS = {k: np.array(v) / np.linalg.norm(v) for k, v in _RAW_DIRECTIONS.items()}

#: Magnitude of the S4 default, so the sweep can be read in units of "the
#: mis-specification the loop was built with".
NOMINAL_MAGNITUDE = float(np.linalg.norm(_RAW_DIRECTIONS["nominal"]))


@dataclass(frozen=True)
class TwinFidelity:
    """How wrong the twin is, as multipliers on the truth plus what it can see.

    Two independent knobs, and keeping them independent is the point. S4
    reported an ablation called `perfect()` that set both, which turned out to
    be a mislabel twice over: the orientation knob was never plumbed through to
    `Twin.sync`, so it did nothing at all, and the twin's object *positions*
    come from the camera under every setting, so no configuration here has ever
    been perfect. The name is now `exact_physics`, which is what it is.
    """
    cube_mass: float = 1.12
    cube_friction: float = 0.85
    joint_damping: float = 1.10
    known_yaw: bool = False        # may the twin be told each cube's true yaw?

    # ------------------------------------------------------------ constructors

    @staticmethod
    def exact_physics(known_yaw: bool = False) -> "TwinFidelity":
        """True mass, friction and damping. Object positions still come from the
        camera, because that is the sensor, not the model."""
        return TwinFidelity(1.0, 1.0, 1.0, known_yaw=known_yaw)

    @staticmethod
    def along(direction: str, magnitude: float,
              known_yaw: bool = False) -> "TwinFidelity":
        """A twin wrong by `magnitude` in log space, along a named direction.

        A negative magnitude is the mirror image: where the default twin thinks
        cubes are heavier and more slippery than they are, a negative one thinks
        they are lighter and grippier. That sign is the whole experiment -- if
        only one sign helps, what helps is the twin's *bias*; if both do, what
        helps is simply that the twin disagrees.
        """
        m = np.exp(magnitude * DIRECTIONS[direction])
        return TwinFidelity(float(m[0]), float(m[1]), float(m[2]), known_yaw)

    # ------------------------------------------------------------ measures

    def log_error(self) -> np.ndarray:
        return np.log([self.cube_mass, self.cube_friction, self.joint_damping])

    def magnitude(self) -> float:
        """Distance from the truth, in log-parameter space."""
        return float(np.linalg.norm(self.log_error()))

    def signed_magnitude(self, direction: str = "nominal") -> float:
        """Magnitude, signed by which way along `direction` it points."""
        e = self.log_error()
        return float(np.sign(e @ DIRECTIONS[direction] or 1.0) * np.linalg.norm(e))

    def physics_is_exact(self) -> bool:
        return self.magnitude() < 1e-12

    def describe(self) -> str:
        if self.physics_is_exact():
            return f"exact physics, yaw {'known' if self.known_yaw else 'unseen'}"
        return (f"m x{self.cube_mass:.2f} mu x{self.cube_friction:.2f} "
                f"b x{self.joint_damping:.2f} (|e|={self.magnitude():.2f}), "
                f"yaw {'known' if self.known_yaw else 'unseen'}")


@dataclass
class Prediction:
    """What the twin says will happen if this plan is executed."""
    clean: bool
    goal_ok: bool
    collateral_m: float
    reason: str
    step_index: int
    plan: planning.Plan

    def summary(self) -> str:
        return (f"{self.plan.label}: goal={'y' if self.goal_ok else 'n'} "
                f"damage={self.collateral_m*1000:.0f}mm reason={self.reason}")


class Twin:
    """A second copy of the world that the planner is allowed to experiment in.

    One twin is built per layout and reused for every task and every candidate
    plan, because compiling the Panda and building a `mink` configuration costs
    far more than a rollout does.
    """

    def __init__(self, scene: SceneSpec, fidelity: TwinFidelity = TwinFidelity()):
        self.fidelity = fidelity
        self.scene = scene
        self.model, _ = build(scene)
        self._mis_specify()
        self.data = mujoco.MjData(self.model)
        self.robot = Panda(self.model, self.data)
        self.robot.home()
        self._qadr = {o.name: self.model.joint(f"{o.name}_free").qposadr[0]
                      for o in scene.objects}
        self._vadr = {o.name: self.model.joint(f"{o.name}_free").dofadr[0]
                      for o in scene.objects}
        self.n_rollouts = 0

    def _mis_specify(self) -> None:
        f = self.fidelity
        for o in self.scene.objects:
            bid = self.model.body(o.name).id
            self.model.body_mass[bid] *= f.cube_mass
            self.model.body_inertia[bid] *= f.cube_mass
            gid = self.model.geom(f"g_{o.name}").id
            self.model.geom_friction[gid, 0] *= f.cube_friction
        self.model.dof_damping[:ARM_DOF] *= f.joint_damping
        mujoco.mj_setConst(self.model, mujoco.MjData(self.model))

    def believes_yaw(self) -> bool:
        return self.fidelity.known_yaw

    # ------------------------------------------------------------ state

    def sync(self, view: StateView, arm_q: np.ndarray, ctrl: np.ndarray,
             gripper_q: np.ndarray, yaw: dict[str, float] | None = None) -> None:
        """Set the twin to the world as the robot currently believes it to be.

        Joint angles come straight across -- a real arm has encoders, and
        pretending otherwise would be a different and less interesting kind of
        pessimism. Object poses come from `view`, which in the loop is the
        camera. Velocities are zeroed because a still photograph has none, which
        is correct here: the robot looks while the scene is at rest.
        """
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qpos[:ARM_DOF] = arm_q
        self.data.qpos[ARM_DOF:ARM_DOF + 2] = gripper_q
        self.data.ctrl[:] = ctrl
        for o in self.scene.objects:
            a = self._qadr[o.name]
            if o.name in view:
                p = view.body(o.name).xpos
                self.data.qpos[a:a + 3] = [p[0], p[1], o.half + 1e-3]
            th = 0.0 if (yaw is None or not self.fidelity.known_yaw) \
                else float(yaw.get(o.name, 0.0))
            self.data.qpos[a + 3:a + 7] = [np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)]
        mujoco.mj_forward(self.model, self.data)

    def positions(self) -> dict[str, np.ndarray]:
        return {o.name: self.data.body(o.name).xpos[:2].copy() for o in self.scene.objects}

    # ------------------------------------------------------------ prediction

    def predict(self, candidate: planning.Plan, task) -> Prediction:
        """Roll a plan out here and score it exactly as the benchmark would."""
        snapshot = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy())
        start = self.positions()
        ex = planning.execute(candidate, self.robot, self.model, self.data)
        goal_ok = bool(task.satisfied(self.model, self.data, self.scene))
        damage = task.collateral(self.model, self.data, self.scene, start)
        self.n_rollouts += 1
        pred = Prediction(clean=goal_ok and damage < tasks.COLLATERAL_TOL,
                          goal_ok=goal_ok, collateral_m=damage,
                          reason=ex.reason, step_index=ex.step_index,
                          plan=candidate)
        # Rewind so the next candidate starts from the same belief rather than
        # from the wreckage of the last one.
        self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:] = snapshot
        mujoco.mj_forward(self.model, self.data)
        return pred


# ---------------------------------------------------------------- the loop

@dataclass
class Choice:
    """The plan the loop settled on, and everything it rejected getting there."""
    plan: planning.Plan | None
    prediction: Prediction | None
    tried: list[Prediction]
    confident: bool                 # did the twin actually endorse this plan?

    @property
    def n_sims(self) -> int:
        return len(self.tried)

    @property
    def reason(self) -> str:
        if self.plan is None:
            return "no_plan"
        return "endorsed" if self.confident else "best_effort"


def choose(task, model, view, scene, twin: Twin, rng, arm_q, ctrl, gripper_q,
           budget: int = 4, true_yaw: dict[str, float] | None = None) -> Choice:
    """Simulate before acting: propose, roll out, repair, repeat.

    `true_yaw` is privileged information and exists only for the ablation that
    asks what the twin's ignorance of orientation costs. It is ignored unless
    the twin's fidelity says `known_yaw`, and the caller is expected to pass it
    only in that case. The loop proper never has it: a top-down blob of colour
    says nothing about which way a cube is turned.

    Returns the first plan the twin predicts will cleanly succeed. If the budget
    runs out it returns the least-bad candidate and says so -- `confident` is
    False, which is a real and useful output. A system that knows it is about to
    do something it could not verify can ask for help; one that cannot tell the
    difference cannot.
    """
    first = planning.direct_plan(task, model, view, scene, rng)
    if first is None:
        return Choice(None, None, [], False)

    tried: list[Prediction] = []
    queue: list[planning.Plan] = [first]
    seen: set[str] = set()

    while queue and len(tried) < budget:
        candidate = queue.pop(0)
        key = candidate.describe()
        if key in seen:
            continue
        seen.add(key)

        twin.sync(view, arm_q, ctrl, gripper_q, yaw=true_yaw)
        pred = twin.predict(candidate, task)
        tried.append(pred)
        if pred.clean:
            return Choice(candidate, pred, tried, True)

        queue.extend(planning.repairs(task, model, view, scene, rng, candidate,
                                      pred.reason, pred.collateral_m))

    best = max(tried, key=lambda p: (p.goal_ok, -p.collateral_m)) if tried else None
    return Choice(best.plan if best else None, best, tried, False)
