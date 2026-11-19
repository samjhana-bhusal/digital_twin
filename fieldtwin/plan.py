"""Plans: what the robot intends to do, and how an intention gets repaired.

A plan is a short list of steps. Keeping it as *data* rather than as a function
call is the point of this module -- a plan has to be handed to the twin and run
there before it is handed to the robot, so it must be something you can inspect,
rewrite and compare, not control flow that has already happened.

The repair catalogue below is short, and every entry in it earns its place from
a measured S2 failure mode rather than from imagination. One rule shapes all of
them: the benchmark scores *clean* success, so a repair that fixes the goal by
disturbing a protected object has not fixed anything. That rules out the obvious
move of shoving a neighbour aside to make room for the fingers, and leaves
changing the grasp instead. The scoring rule, not taste, decides what a legal
repair is.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator

import numpy as np

from . import skills, tasks
from .arena import WORKSPACE_X, WORKSPACE_Y, SceneSpec
from .tasks import CLEARANCE, AtObjectSpotGoal, AwayFromGoal, ToSideGoal

PARK_MIN = 0.13            # a cleared blocker must end up at least this far away
LOOSE_GOALS = (ToSideGoal, AwayFromGoal)   # tolerant enough to reach by pushing


# ---------------------------------------------------------------- steps

@dataclass(frozen=True)
class Move:
    """Pick an object up and put it down at `target`."""
    obj: str
    target: tuple[float, float]
    yaw_offset: float = 0.0           # rotate the wrist this far off the natural grasp
    note: str = "goal"

    def describe(self) -> str:
        y = "" if self.yaw_offset == 0.0 else f" @{np.rad2deg(self.yaw_offset):+.0f}deg"
        return f"move {self.obj} -> ({self.target[0]:.3f}, {self.target[1]:.3f}){y} [{self.note}]"

    def run(self, robot, model, data) -> skills.Outcome:
        got = skills.pick(robot, model, data, self.obj, yaw_offset=self.yaw_offset)
        if not got:
            return got
        return skills.place(robot, model, data, self.obj, self.target)


@dataclass(frozen=True)
class Shove:
    """Push an object along the table without grasping it."""
    obj: str
    direction: tuple[float, float]
    distance: float
    note: str = "shove"

    def describe(self) -> str:
        return (f"shove {self.obj} along ({self.direction[0]:+.2f}, "
                f"{self.direction[1]:+.2f}) by {self.distance*100:.0f} cm [{self.note}]")

    def run(self, robot, model, data) -> skills.Outcome:
        return skills.push(robot, model, data, self.obj, self.direction, self.distance)


@dataclass
class Plan:
    steps: list
    label: str = "direct"

    def __len__(self) -> int:
        return len(self.steps)

    def describe(self) -> str:
        return f"{self.label}: " + " ; ".join(s.describe() for s in self.steps)

    def touches(self) -> set[str]:
        return {s.obj for s in self.steps}


@dataclass
class Execution:
    ok: bool
    reason: str
    step_index: int                    # which step failed; -1 if none did
    outcomes: list = field(default_factory=list)


def execute(plan: Plan, robot, model, data) -> Execution:
    """Run a plan on whatever world it is handed -- the twin or the real one.

    Nothing in this function knows which it is. That is deliberate: the twin
    prediction is only worth anything if the twin runs *the same code path* the
    robot will, so any divergence between them comes from the model and not from
    a second implementation of the plan runner.
    """
    outs = []
    for i, step in enumerate(plan.steps):
        out = step.run(robot, model, data)
        outs.append(out)
        if not out.ok:
            return Execution(False, out.reason, i, outs)
    return Execution(True, outs[-1].reason if outs else "empty", -1, outs)


# ---------------------------------------------------------------- placement sampling

def sample_target(task, model, view, scene: SceneSpec, rng,
                  clearance: float = CLEARANCE,
                  avoid: tuple[str, ...] = ()) -> np.ndarray | None:
    """A concrete (x, y) the goal would accept, clear of other objects.

    This is the planner's own sampler rather than `Task.plan_target`, because
    the two want different things from the same goal. The benchmark's version
    asks "is this instruction satisfiable at all?". The planner's version is
    allowed to be picky -- to demand more room, or to steer clear of a
    particular object -- which is what makes re-sampling a usable repair.
    """
    occupied = [np.asarray(view.body(n).xpos[:2]) for n in view.names()
                if n != task.referent]
    occupied += [tasks.landmark_xy(scene, l.name) for l in scene.landmarks if not l.flat]
    avoid_xy = [np.asarray(view.body(n).xpos[:2]) for n in avoid if n in view]
    for q in task.goal.candidates(model, view, scene, task.referent, rng):
        if not (WORKSPACE_X[0] <= q[0] <= WORKSPACE_X[1]
                and WORKSPACE_Y[0] <= q[1] <= WORKSPACE_Y[1]):
            continue
        if any(np.linalg.norm(q - b) < clearance for b in occupied):
            continue
        if any(np.linalg.norm(q - b) < PARK_MIN for b in avoid_xy):
            continue
        return q
    return None


def _park_spot(blocker: str, keep_clear: np.ndarray, view, scene, rng,
               tries: int = 400) -> np.ndarray | None:
    """Somewhere to put a blocker down that is out of the way and not in a new way."""
    others = [np.asarray(view.body(n).xpos[:2]) for n in view.names() if n != blocker]
    pipes = [tasks.landmark_xy(scene, l.name) for l in scene.landmarks if not l.flat]
    for _ in range(tries):
        q = np.array([rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y)])
        if np.linalg.norm(q - keep_clear) < PARK_MIN:
            continue
        if any(np.linalg.norm(q - b) < CLEARANCE for b in others + pipes):
            continue
        return q
    return None


def occupant(target: np.ndarray, task, view, radius: float = 0.075) -> str | None:
    """Whatever is already sitting where we intend to put something."""
    best, best_d = None, radius
    for n in view.names():
        if n == task.referent:
            continue
        d = float(np.linalg.norm(np.asarray(view.body(n).xpos[:2]) - target))
        if d < best_d:
            best, best_d = n, d
    return best


def movable_blocker(task) -> str | None:
    """The one object a task licenses moving besides the referent, if any.

    Only the occupied-spot goal has one. For every other goal the destination's
    occupant is protected, so clearing it would trade a goal failure for
    collateral damage -- the planner has to re-aim instead.
    """
    if isinstance(task.goal, AtObjectSpotGoal):
        return task.goal._other()
    return None


# ---------------------------------------------------------------- planning

def direct_plan(task, model, view, scene, rng) -> Plan | None:
    """One pick-and-place at the first acceptable placement. The S2 behaviour."""
    q = sample_target(task, model, view, scene, rng)
    if q is None and isinstance(task.goal, AtObjectSpotGoal):
        q = np.asarray(task.goal.spot, float)       # occupied on purpose
    if q is None:
        return None
    return Plan([Move(task.referent, (float(q[0]), float(q[1])))], label="direct")


def repairs(task, model, view, scene, rng, failed: Plan,
            reason: str, collateral_m: float) -> Iterator[Plan]:
    """Candidate rewrites of a plan the twin says will not work.

    Ordered by what the twin saw go wrong. The ordering matters more than the
    catalogue: each candidate costs a full simulated episode, so trying the
    repair that addresses the observed failure first is the difference between
    a loop that pays for itself and one that just burns time.
    """
    goal_step = failed.steps[-1]
    target = np.asarray(goal_step.target, float) if isinstance(goal_step, Move) else None

    grasp_side = reason in ("grasp_missed", "approach_unreachable", "lost_on_lift",
                            "descend_unreachable", "object_slipped")
    place_side = reason in ("placed_off_target", "place_unreachable",
                            "transit_unreachable", "dropped_in_transit")
    blocked = collateral_m >= tasks.COLLATERAL_TOL

    # --- clear the blocker, then place -----------------------------------
    # The headline repair, and the only one that changes the *length* of the
    # plan. Legal only where the goal itself licenses moving the occupant.
    blocker = movable_blocker(task)
    if target is not None and blocker is not None and occupant(target, task, view) == blocker:
        park = _park_spot(blocker, target, view, scene, rng)
        if park is not None:
            yield Plan([Move(blocker, (float(park[0]), float(park[1])), note="clear"),
                        replace(goal_step, note="goal")],
                       label="clear-then-place")

    # --- re-aim ----------------------------------------------------------
    # For a goal with room to move, a placement that bulldozed something or that
    # the arm could not reach is a bad *choice*, not an impossible task.
    if (place_side or blocked) and not isinstance(task.goal, AtObjectSpotGoal):
        for attempt in range(2):
            q = sample_target(task, model, view, scene, rng,
                              clearance=CLEARANCE + 0.02 * (attempt + 1))
            if q is not None:
                yield Plan([replace(goal_step, target=(float(q[0]), float(q[1])),
                                    note="re-aimed")], label="re-aim")

    # --- turn the wrist --------------------------------------------------
    # A cube has 90-degree symmetry, so the fingers can straddle it along either
    # of two axes. When a neighbour sits where one pair of fingers wants to go,
    # the other pair is often clear. This is the licensed way to fix a tight
    # grasp: shoving the neighbour aside would fix the grasp and fail the
    # episode on collateral damage.
    if grasp_side and isinstance(goal_step, Move):
        yield Plan([replace(goal_step, yaw_offset=np.pi / 2, note="turned wrist")],
                   label="turn-wrist")

    # --- push instead of pick -------------------------------------------
    # Only where the goal's tolerance is loose enough to absorb a push's
    # imprecision. Pushing the referent is always licensed -- it is the one
    # object the instruction is about.
    if grasp_side and target is not None and isinstance(task.goal, LOOSE_GOALS):
        here = np.asarray(view.body(task.referent).xpos[:2])
        d = target - here
        n = float(np.linalg.norm(d))
        if n > 0.02:
            yield Plan([Shove(task.referent, (float(d[0] / n), float(d[1] / n)),
                              min(n, 0.16), note="push instead")], label="push")
