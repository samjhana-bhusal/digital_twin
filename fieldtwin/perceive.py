"""Seeing the table through a camera instead of reading the simulator.

Why this exists at all: a digital twin that is *initialised from ground truth*
proves nothing. MuJoCo is deterministic, so a twin handed the plant's exact
state and the exact same plan reproduces the plant bit for bit, and
"did the twin predict the outcome?" becomes a tautology rather than a
measurement. A real system does not have ground truth. It has an image.

So both halves of the S4 loop look through this module:

  * **before acting** -- the twin's initial state is built from what the camera
    sees, which means the plan is checked against a world that is slightly wrong
    in exactly the way a real one would be;
  * **after acting** -- the verifier evaluates the goal predicate on estimated
    positions, so a verification failure is a real disagreement rather than a
    restatement of the predicate.

The estimator itself is deliberately simple and written out rather than
imported: segment by chromaticity, take the median pixel of each colour, and
un-project that pixel onto the plane the cube tops lie in. It is the geometric
baseline that the S3 VLM has to beat, and being simple is what makes it a fair
baseline -- it has no parameters fitted to this scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .arena import CUBE_HALF, PALETTE, SceneSpec

MIN_PIXELS = 60            # fewer than this and we decline to report a position
MIN_SATURATION = 0.20      # below this a pixel is table, arm, pipe or shadow
MAX_HUE_DEG = 25.0         # how far a pixel's hue may sit from a known colour

# Colours in the frame that nothing asks about. They still need a label, or
# their pixels get assigned to whichever cube colour happens to be nearest in
# hue -- the green pad is strongly saturated and would otherwise be read as a
# teal cube every single frame.
_BACKGROUND = ("green",)


def _hue_sat(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised hue (degrees) and saturation for an RGB array in [0, 1].

    Hue rather than RGB distance, and that choice is the whole trick. The scene
    lights wash every surface towards white, which in RGB moves a cube a long
    way from its nominal colour -- far enough that matching nominal RGB finds
    almost nothing. Adding white leaves hue untouched, so hue is the invariant
    the sensor actually preserves. Saturation then separates the coloured
    objects from the table, arm and pipe, which all sit below S = 0.12.
    """
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    c = mx - mn
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    safe = np.where(c > 1e-9, c, 1.0)
    h = np.select(
        [c <= 1e-9, mx == r, mx == g],
        [0.0, ((g - b) / safe) % 6.0, (b - r) / safe + 2.0],
        default=(r - g) / safe + 4.0,
    ) * 60.0
    sat = np.where(mx > 1e-9, c / np.where(mx > 1e-9, mx, 1.0), 0.0)
    return h, sat


def _hue_of(rgb) -> float:
    h, _ = _hue_sat(np.asarray(rgb, float).reshape(1, 3))
    return float(h[0])


def _hue_gap(a: np.ndarray, b: float) -> np.ndarray:
    """Circular distance in degrees."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


@dataclass
class Detection:
    name: str
    colour: str
    xy: np.ndarray
    pixels: int


class Perception:
    """A fixed camera and the un-projection that turns pixels into table coords.

    The camera pose and field of view are read from the model at look time
    rather than hardcoded, so the same code works for the overhead view, the
    three-quarter `front` view, or the wrist camera.
    """

    def __init__(self, model: mujoco.MjModel, camera: str = "overhead",
                 width: int = 480, height: int = 480,
                 plane_z: float = 2 * CUBE_HALF):
        self.model = model
        self.camera = camera
        self.width, self.height = width, height
        self.plane_z = plane_z
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._cam_id = model.camera(camera).id

    def close(self) -> None:
        self._renderer.close()

    # ------------------------------------------------------------ geometry

    def unproject(self, data, u: float, v: float) -> np.ndarray:
        """Pixel (u, v) -> the point where that ray meets the plane z = plane_z.

        Standard pinhole: the vertical field of view fixes the scale, pixels are
        square, and the camera's world rotation comes straight out of MuJoCo.
        The cube's top face centre sits directly above its body centre, so the
        median pixel of a cube's top face un-projects to the body's (x, y) with
        no perspective bias to correct.
        """
        cam_pos = data.cam_xpos[self._cam_id].copy()
        R = data.cam_xmat[self._cam_id].reshape(3, 3).copy()   # cols: x, y, z axes
        f = np.tan(np.deg2rad(self.model.cam_fovy[self._cam_id]) / 2)
        half_h = self.height / 2
        # camera frame: +x right, +y up, looking down -z
        d_cam = np.array([(u - self.width / 2) / half_h * f,
                          (half_h - v) / half_h * f,
                          -1.0])
        d = R @ d_cam
        if abs(d[2]) < 1e-9:
            raise ValueError("ray is parallel to the table plane")
        t = (self.plane_z - cam_pos[2]) / d[2]
        if t < 0:
            raise ValueError("plane is behind the camera")
        return (cam_pos + t * d)[:2]

    # ------------------------------------------------------------ detection

    def image(self, data) -> np.ndarray:
        self._renderer.update_scene(data, camera=self.camera)
        return self._renderer.render()

    def look(self, data, scene: SceneSpec) -> dict[str, Detection]:
        """Estimate every cube's (x, y). Objects it cannot see are simply absent.

        An absent object is a real answer, not an error: a cube hidden behind the
        arm cannot be verified, and the loop has to cope with being unable to
        check rather than pretend it checked.
        """
        img = self.image(data).astype(np.float64) / 255.0
        hue, sat = _hue_sat(img)
        coloured = sat > MIN_SATURATION

        wanted = {o.colour: o.name for o in scene.objects}
        labels = list(wanted) + [c for c in _BACKGROUND if c not in wanted]
        ref = np.array([_hue_of(PALETTE[c][:3]) for c in labels])
        gaps = np.stack([_hue_gap(hue, h) for h in ref], axis=-1)
        nearest = gaps.argmin(axis=-1)
        closest = gaps.min(axis=-1)

        out: dict[str, Detection] = {}
        for k, colour in enumerate(labels):
            if colour not in wanted:
                continue                                   # background label
            mask = coloured & (nearest == k) & (closest < MAX_HUE_DEG)
            n = int(mask.sum())
            if n < MIN_PIXELS:
                continue
            vs, us = np.nonzero(mask)
            # Median, not mean: one stray speck of the right hue somewhere else
            # in the frame drags a mean across the table and leaves a median
            # sitting on the blob where it belongs.
            u, v = float(np.median(us)), float(np.median(vs))
            out[wanted[colour]] = Detection(wanted[colour], colour,
                                            self.unproject(data, u, v), n)
        return out


# ------------------------------------------------------------------ state view

class _Body:
    __slots__ = ("xpos",)

    def __init__(self, xpos):
        self.xpos = xpos


class StateView:
    """A stand-in for `MjData` that answers `.body(name).xpos` from a dict.

    The task layer's goal predicates take `data` and read body positions out of
    it. Rather than duplicating every predicate for estimated state, the
    estimate is dressed up as the one thing those predicates actually use. The
    predicate code stays single-sourced, which is the whole point: the verifier
    is then demonstrably checking *the same condition* the benchmark scores,
    differing only in where the positions came from.
    """

    def __init__(self, xy: dict[str, np.ndarray], z: float = CUBE_HALF):
        self._b = {k: _Body(np.array([float(v[0]), float(v[1]), z]))
                   for k, v in xy.items()}

    def body(self, name: str) -> _Body:
        try:
            return self._b[name]
        except KeyError:
            raise KeyError(f"{name} was not detected") from None

    def __contains__(self, name: str) -> bool:
        return name in self._b

    def names(self) -> list[str]:
        return list(self._b)


def truth_view(data, scene: SceneSpec) -> StateView:
    """Ground-truth positions in the same wrapper. For ablations only."""
    return StateView({o.name: data.body(o.name).xpos[:2].copy() for o in scene.objects})


def detections_view(det: dict[str, Detection]) -> StateView:
    return StateView({k: v.xy for k, v in det.items()})
