"""Scene composition: a real Franka Panda on a tabletop with manipulable objects.

Built with MuJoCo's MjSpec model-editing API rather than hand-written XML. That
is deliberate: the arm model comes from MuJoCo Menagerie (via robot_descriptions)
and stays exactly as its maintainers ship it, while the table, objects, cameras
and the tool-centre-point site are attached programmatically. Nothing is copied
or forked, so a Menagerie update is picked up for free, and the scene can be
regenerated with different object layouts without touching any file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from robot_descriptions import panda_mj_description as _panda

# Franka hand: the point between the fingertips, along +z of the `hand` frame.
TCP_OFFSET = 0.1034
TABLE_Z = 0.0
# Every light in the scene is scaled by this. See `_panda_spec`.
LIGHT_SCALE = 0.30
CUBE_HALF = 0.022

# Where objects may be placed: inside the Panda's comfortable reach, in front.
WORKSPACE_X = (0.36, 0.62)
WORKSPACE_Y = (-0.26, 0.26)

PALETTE = {
    "red":    (0.78, 0.22, 0.20, 1.0),
    "blue":   (0.20, 0.42, 0.78, 1.0),
    "yellow": (0.88, 0.70, 0.16, 1.0),
    "green":  (0.18, 0.60, 0.34, 1.0),
    "grey":   (0.47, 0.50, 0.54, 1.0),
    "purple": (0.52, 0.28, 0.72, 1.0),
    "orange": (0.95, 0.52, 0.10, 1.0),
    "teal":   (0.10, 0.60, 0.58, 1.0),
}

# Cube colours, in the order extra objects are added. Kept visually separable so
# that a grounder is tested on spatial language rather than on telling two
# similar hues apart -- and "separable" means *as rendered*, not as specified.
# Orange was the fifth colour until the camera was measured: under the scene
# lights the red channel clips, which compresses hue, and orange arrived at the
# sensor 10 degrees from yellow. Teal sits 36 degrees clear of both its
# neighbours (the green pad and the blue cube) in the rendered image.
#
#   rendered hue:  red 3   teal 178   blue 211   purple 278   yellow 60
#   also in frame: green pad 143, and the neutral table and pipe below S=0.12
CUBE_COLOURS = ("red", "blue", "yellow", "purple", "teal")


@dataclass
class ObjectSpec:
    """One manipulable object. `name` is what the task layer refers to."""
    name: str
    colour: str
    pos: tuple[float, float]
    yaw: float = 0.0
    half: float = CUBE_HALF
    mass: float = 0.06
    movable: bool = True


@dataclass
class Landmark:
    """Static scene furniture: referred to in instructions, never picked up."""
    name: str
    colour: str
    pos: tuple[float, float]
    radius: float
    height: float
    flat: bool = False          # a flat pad rather than a standing cylinder


@dataclass
class SceneSpec:
    objects: list[ObjectSpec] = field(default_factory=list)
    landmarks: list[Landmark] = field(default_factory=list)

    def object(self, name: str) -> ObjectSpec:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    def landmark(self, name: str) -> Landmark:
        for l in self.landmarks:
            if l.name == name:
                return l
        raise KeyError(name)


def default_scene() -> SceneSpec:
    """The reference layout, used when nothing is randomised."""
    return SceneSpec(
        objects=[
            ObjectSpec("red_cube", "red", (0.46, 0.10)),
            ObjectSpec("blue_cube", "blue", (0.54, -0.14)),
            ObjectSpec("yellow_cube", "yellow", (0.40, -0.05)),
        ],
        landmarks=[
            Landmark("pipe", "grey", (0.60, 0.20), radius=0.030, height=0.085),
            Landmark("green_pad", "green", (0.44, -0.24), radius=0.055, height=0.001, flat=True),
        ],
    )


def _panda_spec() -> mujoco.MjSpec:
    """Menagerie's Panda, untouched, plus the TCP site and wrist camera we need."""
    spec = mujoco.MjSpec.from_file(_panda.MJCF_PATH)
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Menagerie lights the arm for a close-up; combined with our own key/fill and
    # MuJoCo's headlight that drives every saturated surface to pure white, which
    # destroys exactly the signal the colour segmenter in `perceive.py` needs.
    # Lighting is part of the sensor model here, not decoration.
    for light in spec.lights:
        light.diffuse = [v * LIGHT_SCALE for v in light.diffuse]
    spec.visual.headlight.diffuse = [0.12, 0.12, 0.12]
    spec.visual.headlight.ambient = [0.18, 0.18, 0.18]
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.004, 0.004, 0.004],
                  rgba=[1, 1, 0, 0.0])
    hand.add_camera(name="wrist", pos=[0, 0, 0.02], xyaxes=[0, -1, 0, 1, 0, 0],
                    fovy=58)
    return spec


def arm_model() -> mujoco.MjModel:
    """The arm ALONE, for inverse kinematics.

    IK must not run on the full scene: the cubes have free joints, so a solver
    given the whole model would happily 'reach' a target by sliding a cube
    across the table. Solving on an arm-only model makes that impossible rather
    than merely discouraged.
    """
    return _panda_spec().compile()


def build(scene: SceneSpec) -> tuple[mujoco.MjModel, SceneSpec]:
    """Compose Panda + table + scene objects into one compiled model."""
    spec = _panda_spec()
    wb = spec.worldbody

    # --- table -------------------------------------------------------------
    tex = spec.add_texture(name="tabletex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           width=300, height=300,
                           rgb1=[0.26, 0.28, 0.30], rgb2=[0.31, 0.33, 0.35])
    mat = spec.add_material(name="tablemat", texrepeat=[8, 8], reflectance=0.04)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tabletex"
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[1.2, 1.2, 0.05], pos=[0, 0, TABLE_Z], material="tablemat")

    # --- cameras -----------------------------------------------------------
    # Three-quarter view aimed at (0.5, 0, 0.15): holds the whole arm at the
    # home pose AND the workspace. A straight-on front camera clips the
    # shoulder, because the Panda stands tall when it is not reaching.
    wb.add_camera(name="front", pos=[1.90, -1.15, 1.25],
                  xyaxes=[0.621, 0.783, 0, -0.358, 0.284, 0.889], fovy=48)
    # The sensor camera. High and narrow rather than low and wide: at 1.05 m the
    # arm is nearer the lens than the table is, so it looms over the whole frame
    # and hides every cube. At 2.2 m with a 26-degree field the coverage is the
    # same but the arm subtends a quarter of the angle, and the view is close
    # enough to orthographic that a cube's top face projects over its own centre.
    wb.add_camera(name="overhead", pos=[0.49, 0.0, 2.2],
                  xyaxes=[0, 1, 0, -1, 0, 0], fovy=26)
    wb.add_light(name="key", pos=[0.7, -0.3, 1.5], dir=[-0.2, 0.2, -1],
                 diffuse=[0.8 * LIGHT_SCALE] * 3)
    wb.add_light(name="fill", pos=[0.2, 0.6, 1.2], dir=[0.2, -0.4, -1],
                 diffuse=[0.30 * LIGHT_SCALE, 0.30 * LIGHT_SCALE, 0.32 * LIGHT_SCALE])

    # --- landmarks (static) ------------------------------------------------
    for lm in scene.landmarks:
        rgba = list(PALETTE[lm.colour])
        if lm.flat:
            rgba[3] = 0.65
        wb.add_geom(
            name=f"lm_{lm.name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[lm.radius, lm.height / 2 if not lm.flat else 0.0015, 0],
            pos=[lm.pos[0], lm.pos[1], (lm.height / 2) if not lm.flat else 0.0015],
            rgba=rgba,
            contype=0 if lm.flat else 1, conaffinity=0 if lm.flat else 1,
        )

    # --- manipulable objects ----------------------------------------------
    for ob in scene.objects:
        body = wb.add_body(name=ob.name, pos=[ob.pos[0], ob.pos[1], ob.half + 1e-3],
                           quat=[np.cos(ob.yaw / 2), 0, 0, np.sin(ob.yaw / 2)])
        body.add_freejoint(name=f"{ob.name}_free")
        body.add_geom(name=f"g_{ob.name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[ob.half] * 3, rgba=list(PALETTE[ob.colour]),
                      mass=ob.mass, friction=[1.0, 0.01, 0.001],
                      condim=4)
        body.add_site(name=f"s_{ob.name}", size=[0.003] * 3, rgba=[0, 0, 0, 0])

    model = spec.compile()
    return model, scene


def random_scene(rng: np.random.Generator, n_cubes: int = 3,
                 min_sep: float = 0.085) -> SceneSpec:
    """A randomised layout with non-overlapping objects, for benchmarking.

    `min_sep` is the difficulty dial. At 0.085 m the gripper fits around any
    cube without touching its neighbours. Near 0.060 m it does not, and a
    top-down pick starts failing on the approach -- which is the regime where
    clearing a neighbour first actually matters.
    """
    base = default_scene()
    if n_cubes > len(CUBE_COLOURS):
        raise ValueError(f"at most {len(CUBE_COLOURS)} cubes")
    colours = list(CUBE_COLOURS[:n_cubes])
    blocked = [np.array(l.pos) for l in base.landmarks]
    placed: list[np.ndarray] = []
    objs = []
    for c in colours:
        for _ in range(400):
            p = np.array([rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y)])
            if all(np.linalg.norm(p - q) >= min_sep for q in placed + blocked):
                placed.append(p)
                objs.append(ObjectSpec(f"{c}_cube", c, (float(p[0]), float(p[1])),
                                       yaw=float(rng.uniform(-0.4, 0.4))))
                break
        else:
            raise RuntimeError("could not place all objects without overlap")
    return SceneSpec(objects=objs, landmarks=base.landmarks)
