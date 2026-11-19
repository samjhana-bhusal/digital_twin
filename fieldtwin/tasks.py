"""Instructions, and what it means to have carried one out.

A task is two separable things:

  * a **reference** -- which object the instruction is talking about, and
  * a **goal** -- a predicate on simulator state that says whether the task is
    done.

Keeping them apart is what makes the later stages measurable. The reference has
a ground-truth answer computed from simulator state, so when a VLM is dropped in
to resolve references from camera images (S3) its accuracy can be scored against
that answer. The goal is a predicate rather than a human judgement, so success
is never a matter of opinion.

Two rules the generator enforces, because without them a benchmark measures its
own sloppiness:

  * **No ambiguous references.** If "the cube left of the pipe" matches two
    cubes, the instruction is ill-posed and is not generated. A reference must
    beat its runner-up by `MIN_MARGIN` metres.
  * **No already-satisfied goals.** If the red cube is already on the pad, "put
    the red cube on the pad" scores a success for doing nothing. Those are not
    generated either.

Frame of reference: the robot stands at the origin facing +x, so **left is +y**
and **right is -y**, and "near"/"far" are measured from the robot's base. Every
spatial word in this module means it from the robot's point of view, which is
stated here once rather than argued about later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arena import CUBE_HALF, WORKSPACE_X, WORKSPACE_Y, SceneSpec

MIN_MARGIN = 0.055          # a reference must win by this much, in metres
PAD_TOL = 0.045             # "on the pad" means within this of its centre
SPOT_TOL = 0.030            # "where the blue cube is" is a tighter claim
COLLATERAL_TOL = 0.030      # an object this far off its start counts as disturbed
AWAY_DIST = 0.26            # "away from X" means at least this far from X
NEXT_TO_MAX = 0.115         # "next to Y" means centres no further apart than this
NEXT_TO_MIN = 0.058         # ... and no closer, or they are interpenetrating
CLEARANCE = 0.080           # minimum centre distance to any other object
SIDE_HALF = 0.10            # "to the left side" means |y| beyond this


# ---------------------------------------------------------------- helpers

def obj_xy(data, name: str) -> np.ndarray:
    return data.body(name).xpos[:2].copy()


def landmark_xy(scene: SceneSpec, name: str) -> np.ndarray:
    return np.asarray(scene.landmark(name).pos, float)


def _unique_by(scores: dict[str, float], largest: bool) -> tuple[str | None, float]:
    """Pick the winner and report how far clear of the runner-up it is."""
    if len(scores) < 2:
        return (next(iter(scores)), np.inf) if scores else (None, 0.0)
    order = sorted(scores.items(), key=lambda kv: kv[1], reverse=largest)
    return order[0][0], abs(order[0][1] - order[1][1])


# ---------------------------------------------------------------- references

@dataclass(frozen=True)
class ColourRef:
    """"the red cube" -- unambiguous by construction, one cube per colour."""
    colour: str

    def phrase(self) -> str:
        return f"the {self.colour} cube"

    def resolve(self, model, data, scene) -> tuple[str | None, float]:
        name = f"{self.colour}_cube"
        return (name, np.inf) if any(o.name == name for o in scene.objects) else (None, 0.0)


@dataclass(frozen=True)
class ExtremeToLandmarkRef:
    """"the cube nearest the pipe" / "the cube farthest from the pipe"."""
    landmark: str
    nearest: bool

    def phrase(self) -> str:
        word = "nearest" if self.nearest else "farthest from"
        return f"the cube {word} the {self.landmark.replace('_', ' ')}"

    def resolve(self, model, data, scene):
        p = landmark_xy(scene, self.landmark)
        d = {o.name: float(np.linalg.norm(obj_xy(data, o.name) - p)) for o in scene.objects}
        return _unique_by(d, largest=not self.nearest)


@dataclass(frozen=True)
class SideOfLandmarkRef:
    """"the cube to the left of the pipe" -- exactly one cube may be on that side."""
    landmark: str
    side: str                      # "left" | "right"

    def phrase(self) -> str:
        return f"the cube to the {self.side} of the {self.landmark.replace('_', ' ')}"

    def resolve(self, model, data, scene):
        p = landmark_xy(scene, self.landmark)
        sign = 1.0 if self.side == "left" else -1.0
        on_side = {o.name: sign * (obj_xy(data, o.name)[1] - p[1]) for o in scene.objects}
        winners = {k: v for k, v in on_side.items() if v > 0}
        if len(winners) != 1:
            return None, 0.0           # zero or several: the phrase is ill-posed
        name = next(iter(winners))
        others = [v for k, v in on_side.items() if k != name]
        return name, float(winners[name] - max(others)) if others else np.inf


@dataclass(frozen=True)
class SideOfObjectRef:
    """"the cube to the left of the blue cube".

    Object-relative rather than landmark-relative. The landmarks sit at the
    edges of the workspace, so "left of the pipe" is satisfied by all three
    cubes or by none, and almost never by exactly one. Anchoring on another
    cube splits the remaining two cleanly most of the time.
    """
    other_colour: str
    side: str                      # "left" | "right"

    def phrase(self) -> str:
        return f"the cube to the {self.side} of the {self.other_colour} cube"

    def resolve(self, model, data, scene):
        anchor_name = f"{self.other_colour}_cube"
        if not any(o.name == anchor_name for o in scene.objects):
            return None, 0.0
        p = obj_xy(data, anchor_name)
        sign = 1.0 if self.side == "left" else -1.0
        scores = {o.name: sign * (obj_xy(data, o.name)[1] - p[1])
                  for o in scene.objects if o.name != anchor_name}
        winners = {k: v for k, v in scores.items() if v > 0}
        if len(winners) != 1:
            return None, 0.0
        name = next(iter(winners))
        others = [v for k, v in scores.items() if k != name]
        return name, float(winners[name] - max(others)) if others else np.inf


@dataclass(frozen=True)
class SuperlativeRef:
    """"the leftmost cube" / "the cube closest to the robot"."""
    axis: str                      # "leftmost" | "rightmost" | "closest" | "farthest"

    def phrase(self) -> str:
        return {"leftmost": "the leftmost cube", "rightmost": "the rightmost cube",
                "closest": "the cube closest to the robot",
                "farthest": "the cube farthest from the robot"}[self.axis]

    def resolve(self, model, data, scene):
        key = {"leftmost": lambda p: p[1], "rightmost": lambda p: -p[1],
               "closest": lambda p: -np.linalg.norm(p), "farthest": lambda p: np.linalg.norm(p)}[self.axis]
        d = {o.name: float(key(obj_xy(data, o.name))) for o in scene.objects}
        return _unique_by(d, largest=True)


REFERENCE_KIND = {
    ColourRef: "colour",
    ExtremeToLandmarkRef: "extreme-to-landmark",
    SideOfLandmarkRef: "side-of-landmark",
    SideOfObjectRef: "side-of-object",
    SuperlativeRef: "superlative",
}


# ---------------------------------------------------------------- goals

@dataclass(frozen=True)
class OnPadGoal:
    landmark: str

    def phrase(self) -> str:
        return f"on the {self.landmark.replace('_', ' ')}"

    def satisfied(self, model, data, scene, name: str) -> bool:
        p = landmark_xy(scene, self.landmark)
        return float(np.linalg.norm(obj_xy(data, name) - p)) < PAD_TOL

    def candidates(self, model, data, scene, name, rng):
        yield landmark_xy(scene, self.landmark)


@dataclass(frozen=True)
class AwayFromGoal:
    landmark: str

    def phrase(self) -> str:
        return f"away from the {self.landmark.replace('_', ' ')}"

    def satisfied(self, model, data, scene, name: str) -> bool:
        p = landmark_xy(scene, self.landmark)
        return float(np.linalg.norm(obj_xy(data, name) - p)) > AWAY_DIST

    def candidates(self, model, data, scene, name, rng):
        p = landmark_xy(scene, self.landmark)
        for _ in range(300):
            q = np.array([rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y)])
            if np.linalg.norm(q - p) > AWAY_DIST + 0.02:
                yield q


@dataclass(frozen=True)
class NextToGoal:
    other_colour: str

    def phrase(self) -> str:
        return f"next to the {self.other_colour} cube"

    def _other(self) -> str:
        return f"{self.other_colour}_cube"

    def satisfied(self, model, data, scene, name: str) -> bool:
        if name == self._other():
            return False
        d = float(np.linalg.norm(obj_xy(data, name) - obj_xy(data, self._other())))
        return NEXT_TO_MIN < d < NEXT_TO_MAX

    def candidates(self, model, data, scene, name, rng):
        c = obj_xy(data, self._other())
        mid = 0.5 * (NEXT_TO_MIN + NEXT_TO_MAX)
        for _ in range(300):
            a = rng.uniform(-np.pi, np.pi)
            yield c + mid * np.array([np.cos(a), np.sin(a)])


@dataclass(frozen=True)
class ToSideGoal:
    side: str                      # "left" | "right"

    def phrase(self) -> str:
        return f"to the {self.side} side of the table"

    def satisfied(self, model, data, scene, name: str) -> bool:
        y = obj_xy(data, name)[1]
        return y > SIDE_HALF if self.side == "left" else y < -SIDE_HALF

    def candidates(self, model, data, scene, name, rng):
        lo, hi = (SIDE_HALF + 0.03, WORKSPACE_Y[1]) if self.side == "left" \
            else (WORKSPACE_Y[0], -SIDE_HALF - 0.03)
        for _ in range(300):
            yield np.array([rng.uniform(*WORKSPACE_X), rng.uniform(lo, hi)])


@dataclass(frozen=True)
class AtObjectSpotGoal:
    """"put the red cube where the blue cube is".

    The destination is occupied by construction, so a single pick-and-place
    cannot succeed: the blocker has to be moved out of the way first. This is
    the class of task the whole project is aimed at -- a plan that looks fine
    until you simulate it, and that a one-shot scripted pipeline gets wrong
    without ever noticing.

    The spot is captured when the task is generated, not read live, because the
    blocker moves during execution and the goal must not move with it.
    """
    other_colour: str
    spot: tuple[float, float]
    requires_clearing: bool = True

    def phrase(self) -> str:
        return f"where the {self.other_colour} cube is"

    def _other(self) -> str:
        return f"{self.other_colour}_cube"

    def satisfied(self, model, data, scene, name: str) -> bool:
        if name == self._other():
            return False
        return float(np.linalg.norm(obj_xy(data, name) - np.asarray(self.spot))) < SPOT_TOL

    def candidates(self, model, data, scene, name, rng):
        yield np.asarray(self.spot, float)


GOAL_KIND = {OnPadGoal: "on-pad", AwayFromGoal: "away-from",
             NextToGoal: "next-to", ToSideGoal: "to-side",
             AtObjectSpotGoal: "at-occupied-spot"}


# ---------------------------------------------------------------- tasks

@dataclass
class Task:
    instruction: str
    reference: object
    goal: object
    referent: str                  # ground truth, from simulator state
    margin: float                  # how clear the reference was of its runner-up

    @property
    def ref_kind(self) -> str:
        return REFERENCE_KIND[type(self.reference)]

    @property
    def goal_kind(self) -> str:
        return GOAL_KIND[type(self.goal)]

    @property
    def kind(self) -> str:
        return f"{self.ref_kind}/{self.goal_kind}"

    def satisfied(self, model, data, scene) -> bool:
        return self.goal.satisfied(model, data, scene, self.referent)

    def protected(self, scene) -> list[str]:
        """Objects the instruction did not license moving.

        Knocking the rest of the scene around while completing a task is a
        failure even when the goal predicate ends up true -- on real hardware it
        is the difference between doing the job and making a mess. The one
        exception is the blocker in an occupied-spot task, which has to move for
        the task to be possible at all.
        """
        exempt = {self.referent}
        if isinstance(self.goal, AtObjectSpotGoal):
            exempt.add(self.goal._other())
        return [o.name for o in scene.objects if o.name not in exempt]

    def collateral(self, model, data, scene, start_xy: dict[str, np.ndarray]) -> float:
        """Largest distance any protected object was shifted from its start."""
        worst = 0.0
        for name in self.protected(scene):
            worst = max(worst, float(np.linalg.norm(obj_xy(data, name) - start_xy[name])))
        return worst

    def plan_target(self, model, data, scene, rng) -> np.ndarray | None:
        """Turn the goal predicate into one concrete, placeable (x, y).

        Samples candidate placements the goal would accept and returns the first
        that is inside the workspace and clear of everything else. Returning
        None is a real answer: it means the instruction cannot be satisfied in
        this layout, which the benchmark must count separately from a failure
        of the arm.
        """
        if getattr(self.goal, "requires_clearing", False):
            # The point of these tasks is that the destination is occupied.
            # Only the workspace bound applies.
            blockers = []
        else:
            blockers = [obj_xy(data, o.name) for o in scene.objects
                        if o.name != self.referent]
            blockers += [landmark_xy(scene, l.name) for l in scene.landmarks
                         if not l.flat]
        for q in self.goal.candidates(model, data, scene, self.referent, rng):
            if not (WORKSPACE_X[0] <= q[0] <= WORKSPACE_X[1]
                    and WORKSPACE_Y[0] <= q[1] <= WORKSPACE_Y[1]):
                continue
            if any(np.linalg.norm(q - b) < CLEARANCE for b in blockers):
                continue
            return q
        return None


def _all_references(scene: SceneSpec) -> list:
    refs: list = [ColourRef(o.colour) for o in scene.objects]
    for lm in scene.landmarks:
        for nearest in (True, False):
            refs.append(ExtremeToLandmarkRef(lm.name, nearest))
        for side in ("left", "right"):
            refs.append(SideOfLandmarkRef(lm.name, side))
    for o in scene.objects:
        for side in ("left", "right"):
            refs.append(SideOfObjectRef(o.colour, side))
    refs += [SuperlativeRef(a) for a in ("leftmost", "rightmost", "closest", "farthest")]
    return refs


def _all_goals(scene: SceneSpec, data=None) -> list:
    goals: list = []
    for lm in scene.landmarks:
        if lm.flat:
            goals.append(OnPadGoal(lm.name))
        else:
            goals.append(AwayFromGoal(lm.name))
    goals += [NextToGoal(o.colour) for o in scene.objects]
    goals += [AtObjectSpotGoal(o.colour, (float(obj_xy(data, o.name)[0]),
                                          float(obj_xy(data, o.name)[1])))
              for o in scene.objects]
    goals += [ToSideGoal("left"), ToSideGoal("right")]
    return goals


def enumerate_tasks(model, data, scene: SceneSpec, rng) -> list[Task]:
    """Every well-posed instruction for THIS layout.

    Enumerating per layout, rather than fixing instructions and resampling
    layouts until they fit, is what keeps the benchmark cheap: most spatial
    phrases are ambiguous in most layouts, and throwing away layouts would
    throw away most of them.
    """
    out: list[Task] = []
    for ref in _all_references(scene):
        name, margin = ref.resolve(model, data, scene)
        if name is None or margin < MIN_MARGIN:
            continue                                   # ambiguous or ill-posed
        for goal in _all_goals(scene, data):
            if isinstance(goal, (NextToGoal, AtObjectSpotGoal)) and goal._other() == name:
                continue                               # "put X next to/where X"
            if goal.satisfied(model, data, scene, name):
                continue                               # already done; trivial
            verb = "Put" if isinstance(goal, (OnPadGoal, NextToGoal,
                                              AtObjectSpotGoal)) else "Move"
            task = Task(f"{verb} {ref.phrase()} {goal.phrase()}.",
                        ref, goal, name, float(margin))
            if task.plan_target(model, data, scene, rng) is None:
                continue                               # unsatisfiable here
            out.append(task)
    return out
