"""Invariants the harness must keep. These are the claims every later result
silently depends on, so they are tests rather than things we remember."""

import numpy as np
import pytest

from fieldtwin.bus import BusConfig, ServoBus
from fieldtwin.identify.loss import fitting_twin, noise_sigmas, residuals
from fieldtwin.identify.spectrum import (
    finite_difference_jacobian,
    spectrum_from_jacobian,
)
from fieldtwin.models.params import N_PARAMS, ParamSpace
from fieldtwin.plant import make_plant
from fieldtwin.rollout import free_episode, make_dataset, push_episode


@pytest.fixture(scope="module")
def clean_bus():
    return BusConfig(seed=7).deterministic()


def test_param_roundtrip():
    sp = ParamSpace.from_xml()
    import mujoco

    from fieldtwin.models.params import SCENE_XML
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    theta = sp.nominal * 1.7
    sp.set(m, theta)
    assert np.allclose(sp.get(m), theta)
    assert sp.nominal.size == N_PARAMS


def test_mass_change_rescales_inertia():
    import mujoco

    from fieldtwin.models.params import SCENE_XML
    sp = ParamSpace.from_xml()
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    bid = m.body("link1").id
    i0 = m.body_inertia[bid].copy()
    theta = sp.nominal.copy()
    theta[sp.index("m_link1")] *= 2.0
    sp.set(m, theta)
    assert np.allclose(m.body_inertia[bid], 2.0 * i0)


def test_negative_params_rejected():
    sp = ParamSpace.from_xml()
    import mujoco

    from fieldtwin.models.params import SCENE_XML
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    bad = sp.nominal.copy()
    bad[0] = -1.0
    with pytest.raises(ValueError):
        sp.set(m, bad)


def test_bus_holds_setpoint_through_dropout():
    cfg = BusConfig(dropout_p=1.0, seed=0)   # every packet lost
    bus = ServoBus(cfg, nu=3, nq=6)
    bus.reset(np.array([0.1, 0.2, 0.3]))
    bus.send(0.0, np.array([9.0, 9.0, 9.0]))
    held = bus.effective_ctrl(1.0)
    assert np.allclose(held, [0.1, 0.2, 0.3]), "a lost packet must leave the servo holding"
    assert bus.stats.dropped == 1


def test_bus_latency_delays_application():
    cfg = BusConfig(act_latency_s=0.05, act_jitter_s=0.0, dropout_p=0.0, seed=0)
    bus = ServoBus(cfg, nu=1, nq=1)
    bus.reset(np.zeros(1))
    bus.send(0.0, np.array([1.0]))
    assert bus.effective_ctrl(0.049)[0] == 0.0
    assert bus.effective_ctrl(0.051)[0] == 1.0


def test_encoder_quantises_to_grid():
    cfg = BusConfig(encoder_bits=10, sense_latency_s=0.0)
    bus = ServoBus(cfg, nu=1, nq=1)
    bus.reset()
    bus.record(0.0, np.array([0.123456]), np.zeros(1))
    q, _ = bus.observe(0.0)
    res = cfg.encoder_resolution
    assert abs(q[0] / res - round(q[0] / res)) < 1e-9


def test_perfect_bus_has_no_interface_error():
    cfg = BusConfig.perfect()
    assert cfg.encoder_resolution == 0.0
    assert cfg.act_latency_s == 0.0
    assert cfg.dropout_p == 0.0


def test_pushes_make_contact_and_free_episodes_do_not(clean_bus):
    plant = make_plant(bus_cfg=clean_bus)
    rng = np.random.default_rng(3)
    moved = [push_episode(plant, rng, n_steps=100).made_contact() for _ in range(5)]
    assert all(moved), "every planned push must actually reach the box"
    # free_episode raises internally if it touches the box
    assert free_episode(plant, rng, n_steps=120).box_displacement() == 0.0


def test_loss_is_zero_at_true_parameters_and_deterministic(clean_bus):
    """The central validation: with no interface randomness, a twin holding the
    plant's own parameters must reproduce the data exactly."""
    plant = make_plant(bus_cfg=clean_bus)
    trajs = make_dataset(plant, n_push=3, n_free=1, seed=1, n_steps=80)
    sig = noise_sigmas(clean_bus)
    twin = fitting_twin(data_bus=clean_bus)
    r1 = residuals(twin.space.nominal, trajs, twin, sig)
    r2 = residuals(twin.space.nominal, trajs, twin, sig)
    assert np.allclose(r1, 0.0), "clean twin at true theta must have zero residual"
    assert np.array_equal(r1, r2), "loss must be deterministic or the Jacobian is noise"


def test_wrong_parameters_are_detected(clean_bus):
    plant = make_plant(bus_cfg=clean_bus)
    trajs = make_dataset(plant, n_push=3, n_free=1, seed=1, n_steps=80)
    sig = noise_sigmas(clean_bus)
    twin = fitting_twin(data_bus=clean_bus)
    r_true = residuals(twin.space.nominal, trajs, twin, sig)
    r_bad = residuals(twin.space.nominal * 1.10, trajs, twin, sig)
    assert np.linalg.norm(r_bad) > np.linalg.norm(r_true) + 1.0


def test_tip_friction_is_structurally_unidentifiable(clean_bus):
    """MuJoCo combines contact friction with an elementwise max, so while the
    fingertip's mu sits below the box's it has no effect at all. This is a real
    property of the contact model and R1 must report it as an exact null
    direction rather than as a small eigenvalue."""
    plant = make_plant(bus_cfg=clean_bus)
    sp = plant.space
    i_tip, i_box = sp.index("mu_tip"), sp.index("mu_box")
    assert sp.nominal[i_tip] < sp.nominal[i_box]
    finals = []
    for v in (0.05, 0.45):
        th = sp.nominal.copy()
        th[i_tip] = v
        plant.set_theta(th)
        finals.append(push_episode(plant, np.random.default_rng(11), n_steps=100).box[-1].copy())
    assert np.allclose(finals[0], finals[1])


def test_spectrum_finds_the_null_direction(clean_bus):
    plant = make_plant(bus_cfg=clean_bus)
    trajs = make_dataset(plant, n_push=4, n_free=1, seed=1, n_steps=70)
    sig = noise_sigmas(BusConfig(seed=7))        # realistic sensor noise, per config
    twin = fitting_twin(data_bus=clean_bus)
    sp = twin.space
    x0 = sp.to_log(sp.nominal)
    J = finite_difference_jacobian(
        lambda xl: residuals(sp.from_log(xl), trajs, twin, sig), x0, eps=1e-3
    )
    S = spectrum_from_jacobian(J, sp.names)
    assert S.null_dimension() >= 1
    assert S.effective_rank() < N_PARAMS, "half the parameter space should be weak"
    flattest = S.eigvecs[:, 0]
    assert sp.names[int(np.argmax(np.abs(flattest)))] == "mu_tip"
