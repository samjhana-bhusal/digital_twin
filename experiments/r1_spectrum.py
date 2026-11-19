#!/usr/bin/env python
"""R1 entry point: measure what the data can and cannot identify.

    .venv/bin/python experiments/r1_spectrum.py

Writes the spectrum and a mis-specified-twin error decomposition to results/.
This is the week 3-5 experiment in its simplest complete form; the full study
adds the data-budget sweep and the transfer correlation described in PLAN.md.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fieldtwin.bus import BusConfig
from fieldtwin.identify.loss import fitting_twin, noise_sigmas, residuals
from fieldtwin.identify.spectrum import (
    finite_difference_jacobian,
    spectrum_from_jacobian,
)
from fieldtwin.models.params import perturb
from fieldtwin.plant import make_plant
from fieldtwin.rollout import make_dataset
from fieldtwin.trajlog import save_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    with open(os.path.join(ROOT, "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)

    plant_bus = BusConfig(**cfg["plant_bus"])
    ident = cfg["identify"]

    # Study the spectrum on clean dynamics so the eigenvalues describe parameter
    # identifiability rather than interface noise -- but whiten with the noise a
    # real rig would have, or identifiability comes out flatteringly good.
    study_bus = plant_bus.deterministic()
    sigmas = noise_sigmas(plant_bus if ident["whiten_with_realistic_noise"] else study_bus)

    plant = make_plant(bus_cfg=study_bus)
    ds = cfg["dataset"]
    print(f"generating {ds['n_push']} push + {ds['n_free']} free episodes ...")
    trajs = make_dataset(plant, n_push=ds["n_push"], n_free=ds["n_free"],
                         seed=ds["seed"], n_steps=ds["n_steps"])
    log_path = os.path.join(ROOT, "results", "logs", "r1_dataset.npz")
    save_dataset(log_path, trajs, plant.theta, study_bus, note="R1 identifiability study")
    print(f"  saved {log_path}")

    twin = fitting_twin(data_bus=study_bus)
    space = twin.space
    x0 = space.to_log(space.nominal)

    print("computing finite-difference Jacobian ...")
    J = finite_difference_jacobian(
        lambda xl: residuals(space.from_log(xl), trajs, twin, sigmas),
        x0, eps=ident["fd_eps"],
    )
    S = spectrum_from_jacobian(J, space.names)
    tol = ident["flat_rel_tol"]

    print()
    print(S.report(top=5))
    print()
    print(f"exact null directions      : {S.null_dimension()}")
    print(f"stiff condition number     : {S.stiff_condition_number(tol):.3e}")

    # How a realistically mis-specified twin's error splits across the spectrum
    rng = np.random.default_rng(cfg["twin"]["seed"])
    rows = []
    for _ in range(cfg["twin"]["n_twins"]):
        th = perturb(space, space.nominal, cfg["twin"]["rel_sigma"], rng)
        dlog = space.to_log(th) - space.to_log(space.nominal)
        rows.append(S.decompose_error(dlog, tol))
    flat_frac = np.array([r["flat_fraction"] for r in rows])
    print()
    print(f"over {len(rows)} mis-specified twins at rel_sigma="
          f"{cfg['twin']['rel_sigma']}:")
    print(f"  fraction of parameter error in UNIDENTIFIABLE directions: "
          f"{flat_frac.mean():.3f} +/- {flat_frac.std():.3f}")
    print("  (R1's hypothesis: transfer error tracks this, not total error)")

    out = {
        "eigvals": S.eigvals.tolist(),
        "param_names": list(space.names),
        "effective_rank": S.effective_rank(tol),
        "null_dimension": S.null_dimension(),
        "stiff_condition_number": S.stiff_condition_number(tol),
        "flat_fraction_mean": float(flat_frac.mean()),
        "flat_fraction_std": float(flat_frac.std()),
        "config": cfg,
    }
    res_path = os.path.join(ROOT, "results", "r1_spectrum.json")
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {res_path}")


if __name__ == "__main__":
    main()
