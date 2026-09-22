"""Regression tests against the numbers printed in Hodgkin & Huxley (1952).

These are not unit tests in the usual sense. Each one pins a value that appears
in the paper, so that a change to the model which breaks the reproduction fails
loudly rather than quietly.
"""

import numpy as np
import pytest

from hh1952 import model as M
from hh1952.figures import find_threshold


# --- Constants -------------------------------------------------------------

def test_constants_match_page_520():
    assert (M.CM, M.G_NA, M.G_K, M.G_L) == (1.0, 120.0, 36.0, 0.3)
    assert (M.V_NA, M.V_K, M.V_L) == (-115.0, 12.0, -10.613)


def test_resting_state():
    m, h, n = M.resting_state()
    assert m == pytest.approx(0.0529, abs=5e-4)
    assert h == pytest.approx(0.5961, abs=5e-4)
    assert n == pytest.approx(0.3177, abs=5e-4)


def test_leak_potential_balances_the_resting_current():
    """V_l was chosen to make the total ionic current vanish at V = 0."""
    assert M.ionic_current(0.0, *M.resting_state()) == pytest.approx(0.0, abs=0.01)


def test_temperature_factor():
    assert M.phi(M.T_BASE) == pytest.approx(1.0)
    assert M.phi(16.3) == pytest.approx(3.0)          # Q10 = 3, p. 523
    assert M.phi(18.5) == pytest.approx(3.0 ** 1.22, rel=1e-9)   # 3.8202


# --- Rate constants --------------------------------------------------------

@pytest.mark.parametrize("fn, V", [(M.alpha_n, -10.0), (M.alpha_m, -25.0)])
def test_removable_singularities(fn, V):
    """Both alpha forms are 0/0 at one voltage; the limit must be returned."""
    value = fn(V)
    assert np.isfinite(value)
    assert value == pytest.approx(fn(V - 1e-4), rel=1e-3)
    assert value == pytest.approx(fn(V + 1e-4), rel=1e-3)


def test_rate_constants_are_positive_over_the_physiological_range():
    V = np.linspace(-120, 60, 2000)
    for fn in (M.alpha_n, M.beta_n, M.alpha_m, M.beta_m, M.alpha_h, M.beta_h):
        assert np.all(np.isfinite(fn(V)))
        assert np.all(fn(V) > 0)


def test_steady_states_are_bounded():
    V = np.linspace(-120, 60, 500)
    for name in ("m", "h", "n"):
        y = M.steady_state(name, V)
        assert np.all((y >= 0) & (y <= 1))


def test_inactivation_is_monotonic_and_partly_off_at_rest():
    V = np.linspace(-120, 60, 500)
    h = M.steady_state("h", V)
    assert np.all(np.diff(h) > 0)             # h rises as V rises (hyperpolarising)
    assert 0.5 < M.steady_state("h", 0.0) < 0.7


# --- Fig. 12, p. 525 -------------------------------------------------------

@pytest.mark.parametrize("shock, t_peak", [(90, 0.3), (15, 1.1), (7, 3.4)])
def test_membrane_action_potentials_match_figure_12(shock, t_peak):
    tr = M.membrane_action_potential(shock, T=12.0)
    assert tr.fired()
    assert 100.0 < tr.peak_depolarisation < 112.0
    assert tr.time_to_peak == pytest.approx(t_peak, abs=0.1)


def test_six_millivolts_is_subthreshold():
    """H&H's Fig. 12 shows 7 mV firing and 6 mV not."""
    assert not M.membrane_action_potential(6, T=20.0).fired()
    assert M.membrane_action_potential(7, T=20.0).fired()


def test_threshold_lies_in_the_bracket_the_paper_could_resolve():
    assert 6.0 < find_threshold() < 7.0


# --- p. 528, the propagated action potential -------------------------------

def test_shooting_diverges_in_the_direction_the_paper_describes():
    """Too small a K sends V to +infinity, too large to -infinity (p. 522)."""
    assert M.propagated_shoot(9.0)[0] == "+inf"
    assert M.propagated_shoot(12.0)[0] == "-inf"


def test_conduction_velocity_matches_page_528():
    K = M.find_K(temperature_C=18.5)
    assert K == pytest.approx(10.47, rel=0.01)
    assert M.conduction_velocity(K) == pytest.approx(18.8, rel=0.02)


def test_hodgkin_and_huxleys_own_arithmetic():
    """(10470 * 0.0238 / (2 * 35.4 * 1e-6))^(1/2) cm/sec = 18.8 m/sec."""
    assert M.conduction_velocity(10.47) == pytest.approx(18.8, abs=0.05)


# --- Predictions that were never fitted ------------------------------------

def test_anode_break_excitation():
    """Releasing a hyperpolarisation fires the nerve, with no stimulus (Fig. 22)."""
    released = M.membrane_action_potential(hold_V=30.0, T=25.0)
    assert released.fired()
    control = M.membrane_action_potential(hold_V=0.0, T=25.0)
    assert not control.fired()


def test_hyperpolarisation_removes_inactivation():
    assert M.steady_state("h", 30.0) > 0.98
    assert M.steady_state("h", 0.0) < 0.65


def test_refractory_period_recovers_monotonically():
    """A second shock gives a larger response the longer you wait (Fig. 20)."""
    peaks = []
    for delay in (4, 6, 8, 12, 18):
        tr = M.membrane_action_potential(15, T=delay + 14, shocks=[(delay, 90)])
        peaks.append(tr.depolarisation[tr.t >= delay].max())
    assert np.all(np.diff(peaks) > 0)


# --- Numerics --------------------------------------------------------------

def test_euler_and_adaptive_solver_agree_on_peak_height():
    adaptive = M.membrane_action_potential(15, T=12.0)
    euler = M.membrane_action_potential(15, T=12.0, dt=0.002, method="euler")
    assert euler.peak_depolarisation == pytest.approx(
        adaptive.peak_depolarisation, abs=0.2)


def test_euler_converges_at_first_order():
    reference = M.membrane_action_potential(15, T=12.0, dt=0.001)
    errors = []
    for dt in (0.02, 0.01, 0.005):
        tr = M.membrane_action_potential(15, T=12.0, dt=dt, method="euler")
        errors.append(abs(tr.peak_depolarisation - reference.peak_depolarisation))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(1.7 < r < 2.3 for r in ratios)


def test_voltage_clamp_holds_voltage_and_inactivates_sodium():
    tr = M.voltage_clamp(50, T=10.0)
    assert np.allclose(tr.V, -50.0)
    assert tr.g_Na.max() > 10 * tr.g_Na[-1]        # sodium inactivates
    assert tr.g_K[-1] == pytest.approx(tr.g_K.max(), rel=1e-3)   # potassium does not


def test_timed_shocks_land_on_the_requested_times():
    tr = M.membrane_action_potential(2, T=10.0, shocks=[(5.0, 3.0)])
    before = tr.depolarisation[np.argmin(np.abs(tr.t - 4.99))]
    after = tr.depolarisation[np.argmin(np.abs(tr.t - 5.01))]
    assert after - before == pytest.approx(3.0, abs=0.1)
