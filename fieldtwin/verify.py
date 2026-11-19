"""Checking afterwards whether the instruction was actually carried out.

S2 measured the thing that makes this worth building: in roughly 9% of episodes
the skill reported success and the task had failed. That is not a bug in the
skill. `place` knows it grasped an object, carried it and let go, and from
inside `place` that is indistinguishable from success -- it cannot see that the
cube rolled, or that it was the wrong cube, or that the goal needed something
the skill was never told about. Only something outside the skill can.

So the verifier is deliberately *outside*. It gets the same thing a person
walking past would get: a photograph. It re-evaluates the task's own goal
predicate on positions estimated from that image, which makes a disagreement
between verifier and benchmark a genuine perception failure rather than a
restatement of the predicate in different words.

Three outcomes, not two. A verifier that cannot see the object it was asked
about returns `unsure`, and keeping that separate from "failed" is most of the
value: an honest system distinguishes 'I checked and it is wrong' from 'I could
not check', because those call for different responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import tasks
from .perceive import Perception, StateView
from .tasks import AtObjectSpotGoal, NextToGoal


@dataclass
class Verdict:
    goal_ok: bool | None               # None == could not tell
    collateral_m: float | None
    clean: bool | None
    missing: list[str] = field(default_factory=list)
    n_seen: int = 0

    @property
    def unsure(self) -> bool:
        return self.clean is None

    def label(self) -> str:
        if self.clean is None:
            return "unsure"
        return "success" if self.clean else "failure"


def needed_objects(task) -> list[str]:
    """Which objects have to be visible before a verdict is possible at all."""
    need = [task.referent]
    if isinstance(task.goal, NextToGoal):
        need.append(task.goal._other())      # the goal is relative to it
    # AtObjectSpotGoal needs nothing extra: the spot was fixed when the task was
    # generated, so where the blocker ended up does not change the answer.
    return need


def verify(task, scene, before: dict[str, np.ndarray],
           after: dict[str, np.ndarray]) -> Verdict:
    """Evaluate the goal and the collateral rule on two estimated snapshots.

    Both snapshots come from the camera, which matters for the collateral
    number: comparing an estimated 'after' against a ground-truth 'before' would
    fold the estimator's error into every displacement and make a stationary
    cube look like it had been nudged.
    """
    view = StateView(after)
    need = needed_objects(task)
    missing = [n for n in need if n not in view]
    if missing:
        return Verdict(None, None, None, missing, len(after))

    goal_ok = bool(task.goal.satisfied(None, view, scene, task.referent))

    worst = 0.0
    unseen_protected = []
    for name in task.protected(scene):
        if name not in view or name not in before:
            unseen_protected.append(name)
            continue
        worst = max(worst, float(np.linalg.norm(after[name] - before[name])))
    if unseen_protected:
        # A protected object that cannot be found is the most likely one to have
        # been knocked somewhere. Refusing to score is the honest answer.
        return Verdict(goal_ok, None, None, unseen_protected, len(after))

    return Verdict(goal_ok, worst, goal_ok and worst < tasks.COLLATERAL_TOL,
                   [], len(after))


def look(perception: Perception, robot, data, scene,
         park: bool = True, then_ready: bool = False) -> dict[str, np.ndarray]:
    """Get the arm out of frame, photograph the table, return estimated positions.

    Parking is part of perceiving, not a detail of it: the arm stands between
    its own overhead camera and the table, and a look taken without moving it
    first mostly photographs the robot.

    `then_ready` brings the arm back afterwards, which any look taken *before*
    acting should do -- see `Panda.ready`. The look taken after acting does not
    need it, and skipping it there keeps the final photograph honest about
    where everything ended up.
    """
    if park:
        robot.park_for_view()
    seen = {k: v.xy for k, v in perception.look(data, scene).items()}
    if then_ready:
        robot.ready()
    return seen
