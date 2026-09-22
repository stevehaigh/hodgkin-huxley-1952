"""Reproduction of Hodgkin & Huxley (1952), J Physiol 117:500-544."""

from .model import (  # noqa: F401
    CM, G_K, G_NA, G_L, V_K, V_NA, V_L, T_BASE,
    alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h,
    phi, steady_state, tau, resting_state, ionic_current,
    Trajectory, membrane_action_potential, voltage_clamp,
    propagated_shoot, find_K, conduction_velocity, to_modern,
)

__version__ = "0.1.0"
