"""The Hodgkin-Huxley (1952) model, in the paper's own terms.

Hodgkin, A. L. & Huxley, A. F. (1952). A quantitative description of membrane
current and its application to conduction and excitation in nerve.
*J. Physiol.* **117**, 500-544. doi:10.1113/jphysiol.1952.sp004764

Every constant here is transcribed from the printed paper. Nothing is refitted,
rescaled, or modernised. In particular this module keeps the paper's sign
convention, which is the opposite of the modern one:

    V = E - E_r        (membrane potential minus resting potential)

so a **depolarisation is a negative V**. The paper's figures plot -V upwards.
Use :func:`to_modern` if you want conventional millivolts.

Units follow the paper: mV, msec, uF/cm^2, m.mho/cm^2, uA/cm^2.

Two integrators are available everywhere. ``method="lsoda"`` uses SciPy and is
the default; ``method="euler"`` is a fixed-step forward Euler loop that stands
in for the hand computation of p. 523. They agree on every published number to
at least four significant figures, which the notebook demonstrates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "CM", "G_K", "G_NA", "G_L", "V_K", "V_NA", "V_L", "T_BASE",
    "AXON_RADIUS_CM", "AXIAL_RESISTIVITY",
    "alpha_n", "beta_n", "alpha_m", "beta_m", "alpha_h", "beta_h",
    "phi", "steady_state", "tau", "resting_state", "ionic_current",
    "Trajectory", "membrane_action_potential", "voltage_clamp",
    "propagated_shoot", "find_K", "conduction_velocity", "to_modern",
]

# --- Constants, "Summary of equations and parameters", pp. 518-520 ----------

CM = 1.0        # membrane capacity, uF/cm^2
G_K = 36.0      # \bar{g}_K,  m.mho/cm^2
G_NA = 120.0    # \bar{g}_Na, m.mho/cm^2
G_L = 0.3       # \bar{g}_l,  m.mho/cm^2

V_K = 12.0      # mV, paper's sign convention
V_NA = -115.0   # mV
V_L = -10.613   # mV, chosen so that total ionic current vanishes at V = 0

T_BASE = 6.3    # degC: the temperature the rate-constant tables were built for
Q10 = 3.0       # p. 523, "Temperature differences"

# Axon 17 geometry, used for the conduction velocity on p. 528
AXON_RADIUS_CM = 0.0238     # a, cm
AXIAL_RESISTIVITY = 35.4    # R_2, ohm.cm


# --- Rate constants --------------------------------------------------------
# Each is an empirical function H&H fitted to voltage-clamp measurements.
# The (V + k)/(exp((V + k)/10) - 1) forms have a removable singularity at
# V = -k; _singular supplies the limiting value there rather than a NaN.

def _singular(V, offset, scale):
    """scale * (V + offset) / (exp((V + offset)/10) - 1), safe at V = -offset."""
    V = np.asarray(V, dtype=float)
    x = (V + offset) / 10.0
    small = np.abs(x) < 1e-7          # mask first: np.where evaluates both arms
    denom = np.where(small, 1.0, np.expm1(x))
    out = np.where(small, scale * 10.0, scale * (V + offset) / denom)
    return out[()] if out.ndim == 0 else out


def alpha_n(V):
    """eqn (12), p. 518."""
    return _singular(V, 10.0, 0.01)


def beta_n(V):
    """eqn (13), p. 518."""
    return 0.125 * np.exp(np.asarray(V, dtype=float) / 80.0)


def alpha_m(V):
    """eqn (20), p. 518."""
    return _singular(V, 25.0, 0.1)


def beta_m(V):
    """eqn (21), p. 518."""
    return 4.0 * np.exp(np.asarray(V, dtype=float) / 18.0)


def alpha_h(V):
    """eqn (23), p. 518."""
    return 0.07 * np.exp(np.asarray(V, dtype=float) / 20.0)


def beta_h(V):
    """eqn (24), p. 518."""
    return 1.0 / (np.exp((np.asarray(V, dtype=float) + 30.0) / 10.0) + 1.0)


_RATES = {"n": (alpha_n, beta_n), "m": (alpha_m, beta_m), "h": (alpha_h, beta_h)}


def phi(temperature_C: float) -> float:
    """Temperature factor, p. 523.

    "To obtain the action potential at some other temperature T degC the direct
    method would be to multiply all alpha's and beta's by a factor
    phi = 3^((T - 6.3)/10), this being correct for a Q_10 of 3."
    """
    return Q10 ** ((temperature_C - T_BASE) / 10.0)


def steady_state(which: str, V):
    """n_inf, m_inf or h_inf: eqns (9), (22), (25). Independent of temperature."""
    a, b = _RATES[which]
    return a(V) / (a(V) + b(V))


def tau(which: str, V, temperature_C: float = T_BASE):
    """tau_n, tau_m or tau_h in msec: eqn (10) and its analogues."""
    a, b = _RATES[which]
    return 1.0 / (phi(temperature_C) * (a(V) + b(V)))


def resting_state(V: float = 0.0) -> tuple[float, float, float]:
    """(m, h, n) at rest. phi cancels, so this is temperature-independent."""
    return tuple(float(steady_state(k, V)) for k in ("m", "h", "n"))


def ionic_current(V, m, h, n):
    """The three ionic terms of eqn (26), in uA/cm^2."""
    return (
        G_K * n ** 4 * (V - V_K)
        + G_NA * m ** 3 * h * (V - V_NA)
        + G_L * (V - V_L)
    )


def _gates(V, m, h, n, ph):
    """eqns (15), (16), (7): dm/dt, dh/dt, dn/dt."""
    return (
        ph * (alpha_m(V) * (1 - m) - beta_m(V) * m),
        ph * (alpha_h(V) * (1 - h) - beta_h(V) * h),
        ph * (alpha_n(V) * (1 - n) - beta_n(V) * n),
    )


# --- Trajectory ------------------------------------------------------------

@dataclass
class Trajectory:
    """One numerical solution, with the derived quantities H&H plotted."""

    t: np.ndarray       # msec
    V: np.ndarray       # mV, paper's sign convention (depolarisation negative)
    m: np.ndarray
    h: np.ndarray
    n: np.ndarray

    @property
    def depolarisation(self):
        """-V, i.e. what the paper's figures plot upwards."""
        return -self.V

    @property
    def g_K(self):
        """Potassium conductance, m.mho/cm^2."""
        return G_K * self.n ** 4

    @property
    def g_Na(self):
        """Sodium conductance, m.mho/cm^2."""
        return G_NA * self.m ** 3 * self.h

    @property
    def g_total(self):
        """g_Na + g_K + g_l, the quantity compared with impedance data in Fig. 16."""
        return self.g_Na + self.g_K + G_L

    @property
    def I_Na(self):
        return self.g_Na * (self.V - V_NA)

    @property
    def I_K(self):
        return self.g_K * (self.V - V_K)

    @property
    def I_l(self):
        return G_L * (self.V - V_L)

    @property
    def peak_depolarisation(self) -> float:
        return float(self.depolarisation.max())

    @property
    def time_to_peak(self) -> float:
        return float(self.t[np.argmax(self.depolarisation)])

    def fired(self, threshold: float = 50.0) -> bool:
        """Did this solution spike, rather than decay back towards rest?"""
        return self.peak_depolarisation > threshold


# --- Membrane action potential, eqn (26) with I = 0 ------------------------

def _rhs_membrane(t, y, ph):
    V, m, h, n = y
    dm, dh, dn = _gates(V, m, h, n, ph)
    return [-ionic_current(V, m, h, n) / CM, dm, dh, dn]


def _euler_membrane(y0, T, dt, ph):
    steps = int(round(T / dt))
    y = np.empty((steps + 1, 4))
    y[0] = y0
    for i in range(steps):
        d = _rhs_membrane(0.0, y[i], ph)
        y[i + 1] = y[i] + np.asarray(d) * dt
    return np.arange(steps + 1) * dt, y.T


def membrane_action_potential(depolarisation_mV=0.0, T=20.0, dt=0.01,
                              temperature_C=T_BASE, method="lsoda",
                              hold_V=0.0, shocks=()) -> Trajectory:
    """Solve eqn (26) with I = 0 following an instantaneous shock.

    This is the space-clamped ("membrane") action potential of Figs. 12-14 and
    19-22. ``depolarisation_mV`` is positive for a depolarising shock, matching
    the numbers printed on the paper's curves; V starts at its negative.

    ``hold_V`` is a displacement of V (in the paper's raw convention, so
    *positive* means hyperpolarised) applied for all t < 0, long enough for the
    gates to reach their steady state there. This is the boundary condition of
    Fig. 22, the anode break: ``hold_V=30`` holds -V at -30 mV and then lets go.

    ``shocks`` is a sequence of ``(time_msec, depolarisation_mV)`` pairs,
    each applied as an instantaneous step, for the paired-shock experiments
    of Figs. 20 and 21.
    """
    ph = phi(temperature_C)
    y0 = [float(hold_V) - abs(float(depolarisation_mV)), *resting_state(hold_V)]

    if method == "euler":
        if shocks:
            raise NotImplementedError("timed shocks require method='lsoda'")
        t, (V, m, h, n) = _euler_membrane(y0, T, dt, ph)
        return Trajectory(t, V, m, h, n)

    # Integrate in segments so each shock lands exactly on a segment boundary.
    edges = [0.0, *sorted(float(s[0]) for s in shocks if 0.0 < s[0] < T), T]
    jumps = {float(s[0]): abs(float(s[1])) for s in shocks}
    ts, ys, y = [], [], list(y0)
    for start, stop in zip(edges[:-1], edges[1:]):
        if start in jumps:
            y[0] -= jumps[start]
        t_eval = np.arange(start, stop + dt / 2, dt)
        t_eval = t_eval[t_eval <= stop]     # arange can overshoot by a rounding step
        sol = solve_ivp(_rhs_membrane, (start, stop), y, args=(ph,),
                        t_eval=t_eval, method="LSODA", rtol=1e-9, atol=1e-11)
        ts.append(sol.t)
        ys.append(sol.y)
        y = list(sol.y[:, -1])
    t = np.concatenate(ts)
    V, m, h, n = np.concatenate(ys, axis=1)
    return Trajectory(t, V, m, h, n)


# --- Voltage clamp, Figs. 3 and 6 -----------------------------------------

def _rhs_clamp(t, y, V, ph):
    return list(_gates(V, *y, ph))


def voltage_clamp(clamp_mV, T=10.0, dt=0.01, temperature_C=T_BASE) -> Trajectory:
    """Hold V fixed and let m, h and n run, as under the voltage clamp.

    ``clamp_mV`` is the depolarisation in mV (positive), so V is held at its
    negative throughout.
    """
    V = -abs(float(clamp_mV))
    t_eval = np.arange(0.0, T + dt / 2, dt)
    sol = solve_ivp(_rhs_clamp, (0.0, T), list(resting_state()),
                    args=(V, phi(temperature_C)), t_eval=t_eval,
                    method="LSODA", rtol=1e-9, atol=1e-11)
    m, h, n = sol.y
    return Trajectory(sol.t, np.full_like(sol.t, V), m, h, n)


# --- Propagated action potential, eqns (30)-(31) ---------------------------

def _rhs_propagated(t, y, K, ph):
    V, dV, m, h, n = y
    dm, dh, dn = _gates(V, m, h, n, ph)
    # eqn (31): d2V/dt2 = K (C_M dV/dt + I_i)
    return [dV, K * (CM * dV + ionic_current(V, m, h, n)), dm, dh, dn]


def propagated_shoot(K, T=6.0, temperature_C=18.5, V0=-0.1,
                     upper=40.0, lower=-300.0, dt=0.005, method="lsoda"):
    """Integrate eqn (31) for one guess of K and report which way it diverges.

    From p. 522: "It is necessary to guess a value of theta, insert it in eqn.
    (30) and carry out the numerical solution starting from the resting state
    at the foot of the action potential. It is then found that V goes off
    towards either +infinity or -infinity, according as the guessed theta was
    too small or too large."

    Returns ``(outcome, trajectory)`` where outcome is ``"+inf"``, ``"-inf"``
    or ``"bounded"``. The bracketing convention follows the paper: a value of K
    that is too small sends V to +infinity.
    """
    ph = phi(temperature_C)
    y0 = [V0, 0.0, *resting_state()]

    def hit_upper(t, y, *a):
        return y[0] - upper

    def hit_lower(t, y, *a):
        return y[0] - lower

    hit_upper.terminal = hit_lower.terminal = True

    sol = solve_ivp(_rhs_propagated, (0.0, T), y0, args=(float(K), ph),
                    events=(hit_upper, hit_lower), method="LSODA",
                    rtol=1e-8, atol=1e-10, max_step=0.05,
                    t_eval=np.arange(0.0, T + dt / 2, dt))
    V, dV, m, h, n = sol.y
    traj = Trajectory(sol.t, V, m, h, n)
    if sol.t_events[0].size:
        return "+inf", traj
    if sol.t_events[1].size:
        return "-inf", traj
    return "bounded", traj


def find_K(lo=9.0, hi=12.0, iterations=40, **kwargs) -> float:
    """Bisect on H&H's own divergence criterion to find K in eqn (31).

    ``lo`` must diverge to +infinity and ``hi`` to -infinity. The paper reports
    K = 10.47 msec^-1 for axon 17 at 18.5 degC (p. 528).
    """
    if propagated_shoot(lo, **kwargs)[0] != "+inf":
        raise ValueError(f"lower bracket K={lo} does not diverge to +infinity")
    if propagated_shoot(hi, **kwargs)[0] != "-inf":
        raise ValueError(f"upper bracket K={hi} does not diverge to -infinity")
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if propagated_shoot(mid, **kwargs)[0] == "-inf":
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def conduction_velocity(K, radius_cm=AXON_RADIUS_CM,
                        resistivity=AXIAL_RESISTIVITY) -> float:
    """theta in m/sec, from K = 2 R_2 theta^2 C_M / a (p. 528).

    H&H's own arithmetic for axon 17:
        (10470 * 0.0238 / (2 * 35.4 * 1e-6))^(1/2) cm/sec = 18.8 m/sec
    """
    theta_cm_per_sec = np.sqrt(
        (K * 1000.0) * radius_cm / (2 * resistivity * CM * 1e-6)
    )
    return float(theta_cm_per_sec / 100.0)


# --- Convenience -----------------------------------------------------------

def to_modern(V, resting_potential_mV=-65.0):
    """Convert the paper's V to a conventional membrane potential in mV.

    The paper's V is (E - E_r) with depolarisation negative, so the modern
    membrane potential is E_r - V. H&H never fixed a value for E_r, since every
    voltage in the paper is relative; -65 mV is the figure usually supplied
    when the model is restated in modern form.
    """
    return resting_potential_mV - np.asarray(V, dtype=float)
