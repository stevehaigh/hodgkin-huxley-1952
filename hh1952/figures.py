"""Figure builders for the Hodgkin-Huxley (1952) reproduction.

Every figure here is computed from :mod:`hh1952.model`, which contains nothing
but the constants printed in the paper. No figure reproduces scanned artwork,
and none contains experimental data: where H&H plotted measurements, this
module plots only the theoretical curve they drew through them.

Colour is used sparingly and always by entity, never by rank:

    sodium / m      orange    #eb6834
    potassium / n   blue      #2a78d6
    leak / h        aqua      #1baf7a
    membrane potential        ink (black)

Those three hues are slots 1-3 of a palette validated for colour-vision
deficiency at all pairs (worst CVD Delta E 9.2, normal-vision 24.0). Aqua falls
below 3:1 contrast on a white surface, so every series is also direct-labelled
or legended rather than identified by colour alone.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import model as M

__all__ = [
    "SURFACE", "INK", "INK_2", "MUTED", "NA", "K", "LEAK", "GATE",
    "use_paper_style", "equivalent_circuit", "rate_constants",
    "steady_states", "time_constants", "clamp_conductances",
    "membrane_action_potentials", "action_potential_with_conductances",
    "find_threshold",
    "gating_variables", "propagated_shooting", "refractory_period",
    "threshold", "anode_break", "integrator_comparison",
]

# --- Palette ---------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"

NA = "#eb6834"      # sodium, and the m gate
K = "#2a78d6"       # potassium, and the n gate
LEAK = "#1baf7a"    # leak, and the h gate

GATE = {"m": NA, "h": LEAK, "n": K}


def use_paper_style() -> None:
    """Set rcParams once, for the whole notebook.

    The surface is set explicitly rather than left transparent, so the figures
    stay legible when the notebook is viewed in a dark theme.
    """
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.7,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
    })


def _finish(ax, title=None, xlabel=None, ylabel=None, note=None):
    if title:
        ax.set_title(title, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    if note:
        ax.annotate(note, xy=(1.0, -0.19), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7.5, color=MUTED)
    return ax


def _label(ax, x, y, text, color, dx=0, dy=0, fontsize=8, **kw):
    """Direct label, in the series colour, anchored beside the mark."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=fontsize, **kw)


# --- Fig. 1, the equivalent circuit ---------------------------------------

def equivalent_circuit(ax=None):
    """Redraw the equivalent circuit of Fig. 1, p. 501.

    A diagram, not data: this is the one figure in the notebook that is drawn
    rather than computed. The circuit is what eqn (26) says in pictures.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.grid(False)

    top, bot = 5.0, 1.0
    ax.plot([0.4, 9.6], [top, top], color=INK, lw=1.4)
    ax.plot([0.4, 9.6], [bot, bot], color=INK, lw=1.4)
    ax.text(0.4, top + 0.35, "Outside", color=INK_2, fontsize=9)
    ax.text(0.4, bot - 0.6, "Inside", color=INK_2, fontsize=9)

    def lead(x, y0, y1, color=INK):
        ax.plot([x, x], [y0, y1], color=color, lw=1.2)

    def capacitor(x, label):
        lead(x, top, 3.55)
        ax.plot([x - 0.55, x + 0.55], [3.55, 3.55], color=INK, lw=1.8)
        ax.plot([x - 0.55, x + 0.55], [3.15, 3.15], color=INK, lw=1.8)
        lead(x, 3.15, bot)
        ax.text(x + 0.75, 3.35, label, color=INK_2, fontsize=9, va="center")

    def branch(x, color, r_label, e_label, variable=True):
        lead(x, top, 4.1, color)
        box = mpl.patches.Rectangle((x - 0.28, 3.1), 0.56, 1.0,
                                    fill=False, ec=color, lw=1.5,
                                    joinstyle="round")
        ax.add_patch(box)
        if variable:                      # arrow through the box: a variable resistance
            ax.annotate("", xy=(x + 0.5, 4.15), xytext=(x - 0.5, 3.05),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                        shrinkA=0, shrinkB=0))
        lead(x, 3.1, 2.55, color)
        ax.plot([x - 0.45, x + 0.45], [2.55, 2.55], color=color, lw=1.8)   # long plate
        ax.plot([x - 0.22, x + 0.22], [2.25, 2.25], color=color, lw=1.8)   # short plate
        lead(x, 2.25, bot, color)
        ax.text(x + 0.62, 3.6, r_label, color=color, fontsize=9, va="center")
        ax.text(x + 0.62, 2.4, e_label, color=color, fontsize=9, va="center")

    capacitor(1.5, "$C_M$")
    branch(4.0, NA, "$R_{Na}$", "$E_{Na}$")
    branch(6.4, K, "$R_K$", "$E_K$")
    branch(8.8, LEAK, "$R_l$", "$E_l$", variable=False)

    ax.set_title("Fig. 1 (p. 501). The membrane as a circuit: a capacity in "
                 "parallel\nwith three conductances, two of them variable",
                 color=INK, loc="left", fontsize=10)
    return ax


# --- Figs. 4, 7, 9: the fitted rate constants ------------------------------

def rate_constants(V=None, axes=None):
    """alpha and beta for n, m and h: eqns (12), (13), (20), (21), (23), (24).

    These are the empirical curves H&H drew through their voltage-clamp
    measurements. The measured points themselves are not reproduced here.
    """
    if V is None:
        V = np.linspace(-110, 20, 801)
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)

    panels = [
        ("n", r"$\alpha_n$ (12), $\beta_n$ (13)", M.alpha_n, M.beta_n),
        ("m", r"$\alpha_m$ (20), $\beta_m$ (21)", M.alpha_m, M.beta_m),
        ("h", r"$\alpha_h$ (23), $\beta_h$ (24)", M.alpha_h, M.beta_h),
    ]
    for ax, (name, title, a, b) in zip(axes, panels):
        c = GATE[name]
        ax.plot(V, a(V), color=c, label=rf"$\alpha_{name}$")
        ax.plot(V, b(V), color=c, ls="--", label=rf"$\beta_{name}$")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1e2)
        _finish(ax, title, "$V$ (mV)", "rate constant (msec$^{-1}$)")
        ax.legend(loc="upper left")
    axes[0].figure.suptitle(
        "Figs. 4, 7 and 9: the rate constants H&H fitted to the voltage clamp "
        "(6.3 $^\\circ$C; log scale, they span four decades)",
        color=INK, x=0.005, y=1.06, ha="left", fontsize=10)
    return axes


def steady_states(V=None, ax=None):
    """n_inf, m_inf and h_inf: eqns (9), (22), (25). Figs. 5, 8 and 10."""
    if V is None:
        V = np.linspace(-110, 50, 801)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.4))
    for name, eqn in (("n", "9"), ("m", "22"), ("h", "25")):
        y = M.steady_state(name, V)
        ax.plot(V, y, color=GATE[name], label=rf"${name}_\infty$  eqn ({eqn})")
        i = np.argmax(np.abs(np.gradient(y)))
        _label(ax, V[i], y[i], rf"${name}_\infty$", GATE[name], dx=6, dy=2)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center left")
    return _finish(ax, "Figs. 5, 8 and 10: steady-state gating against membrane potential",
                   "$V$ (mV)   [depolarisation is negative]", "fraction open",
                   note="depolarisation to the left")


def time_constants(V=None, ax=None, temperature_C=M.T_BASE):
    """tau_n, tau_m and tau_h in msec: eqn (10) and its analogues."""
    if V is None:
        V = np.linspace(-110, 50, 801)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.4))
    for name in ("n", "m", "h"):
        y = M.tau(name, V, temperature_C)
        ax.plot(V, y, color=GATE[name], label=rf"$\tau_{name}$")
        i = np.argmax(y)
        _label(ax, V[i], y[i], rf"$\tau_{name}$", GATE[name], dx=5, dy=2)
    ax.legend(loc="upper left")
    return _finish(ax, f"Time constants at {temperature_C} $^\\circ$C: "
                       r"$\tau_x = 1/(\alpha_x + \beta_x)$",
                   "$V$ (mV)", "msec")


# --- Figs. 3 and 6: conductances under voltage clamp ----------------------

def clamp_conductances(clamps=(109, 88, 63, 38, 26, 10), axes=None,
                       T=10.0, temperature_C=M.T_BASE):
    """g_K and g_Na during maintained depolarisations.

    H&H fitted these families by hand, one curve at a time (eqns 11 and 19,
    with the per-clamp parameters of Table 2). Here they fall out of the full
    m, h, n kinetics with no per-curve fitting at all, which is a stronger
    statement than the paper could make.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)
    ax_k, ax_na = axes
    shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(clamps)))
    shades_na = plt.cm.Oranges(np.linspace(0.45, 0.95, len(clamps)))

    for c, sk, sn in zip(clamps, shades, shades_na):
        tr = M.voltage_clamp(c, T=T, temperature_C=temperature_C)
        ax_k.plot(tr.t, tr.g_K, color=sk, lw=1.4)
        ax_na.plot(tr.t, tr.g_Na, color=sn, lw=1.4)
        _label(ax_k, tr.t[-1], tr.g_K[-1], f"{c}", K, dx=3, dy=-3, fontsize=7)
        j = int(np.argmax(tr.g_Na))
        _label(ax_na, tr.t[j], tr.g_Na[j], f"{c}", NA, dx=2, dy=3, fontsize=7)

    _finish(ax_k, "Fig. 3: potassium conductance, $g_K = \\bar{g}_K n^4$",
            "msec", "m.mho/cm$^2$",
            note="labels give the depolarisation in mV")
    _finish(ax_na, "Fig. 6: sodium conductance, $g_{Na} = \\bar{g}_{Na} m^3 h$",
            "msec", "m.mho/cm$^2$",
            note="labels give the depolarisation in mV")
    ax_na.set_xlim(0, min(T, 6.0))
    return axes


# --- Fig. 12: membrane action potentials ----------------------------------

def membrane_action_potentials(shocks=(90, 15, 7, 6), ax=None, T=12.0,
                               temperature_C=M.T_BASE):
    """The upper family of Fig. 12, p. 525: solutions of eqn (26) with I = 0.

    A single quantity plotted four times, so the curves are direct-labelled and
    carry no colour coding.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.8))
    for s in shocks:
        tr = M.membrane_action_potential(s, T=T, temperature_C=temperature_C)
        fired = tr.fired()
        ax.plot(tr.t, tr.depolarisation, color=INK if fired else MUTED,
                lw=1.6 if fired else 1.2)
        j = int(np.argmax(tr.depolarisation)) if fired else len(tr.t) // 8
        _label(ax, tr.t[j], tr.depolarisation[j], f"{s}", INK if fired else MUTED,
               dx=2, dy=4)
    ax.set_xlim(0, 6)
    ax.set_ylim(-20, 118)
    return _finish(ax, "Fig. 12 (p. 525): membrane action potentials, "
                       f"{temperature_C} $^\\circ$C",
                   "msec", "$-V$ (mV)",
                   note="labels give the initial depolarisation in mV; "
                        "6 mV is subthreshold")


def action_potential_with_conductances(depolarisation=15, ax=None, T=12.0,
                                       temperature_C=M.T_BASE):
    """Fig. 16-style: the spike beside the conductances that produce it.

    Two measures of different scale, so two stacked axes rather than one chart
    with two y-scales.
    """
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True,
                               height_ratios=[1.0, 1.0],
                               constrained_layout=True)
    ax_v, ax_g = ax
    tr = M.membrane_action_potential(depolarisation, T=T,
                                     temperature_C=temperature_C)
    ax_v.plot(tr.t, tr.depolarisation, color=INK)
    _finish(ax_v, "Fig. 16: the action potential and the conductances beneath it",
            None, "$-V$ (mV)")

    ax_g.plot(tr.t, tr.g_Na, color=NA, label="$g_{Na}$")
    ax_g.plot(tr.t, tr.g_K, color=K, label="$g_K$")
    j_na, j_k = int(np.argmax(tr.g_Na)), int(np.argmax(tr.g_K))
    _label(ax_g, tr.t[j_na], tr.g_Na[j_na], "$g_{Na}$", NA, dx=4, dy=0)
    _label(ax_g, tr.t[j_k], tr.g_K[j_k], "$g_K$", K, dx=4, dy=0)
    ax_g.legend(loc="upper right")
    _finish(ax_g, None, "msec", "m.mho/cm$^2$")
    ax_v.set_xlim(0, min(T, 8.0))
    return ax


def gating_variables(depolarisation=15, ax=None, T=12.0,
                     temperature_C=M.T_BASE):
    """m, h and n through one action potential."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.4))
    tr = M.membrane_action_potential(depolarisation, T=T,
                                     temperature_C=temperature_C)
    for name, y in (("m", tr.m), ("h", tr.h), ("n", tr.n)):
        ax.plot(tr.t, y, color=GATE[name], label=f"${name}$")
        j = int(np.argmax(np.abs(np.gradient(y))))
        _label(ax, tr.t[j], y[j], f"${name}$", GATE[name], dx=6, dy=4)
    ax.legend(loc="upper right")
    ax.set_xlim(0, min(T, 8.0))
    ax.set_ylim(-0.03, 1.05)
    return _finish(ax, "The three gating variables through one spike",
                   "msec", "fraction open",
                   note="$m$ rises first, $h$ shuts the sodium system, $n$ repolarises")


# --- Figs. 15, 17: the propagated action potential ------------------------

def propagated_shooting(K_star=None, offsets=(0.5, 0.1, 0.02, 0.004), ax=None,
                        T=3.0, temperature_C=18.5):
    """The shooting problem of p. 522, drawn.

    No finite K gives a solution that stays bounded for ever: the correct value
    is a separatrix, and every trajectory eventually peels off it. What the
    bisection converges on is the K whose trajectory follows the true action
    potential *longest* before running away. Bracketing values are drawn either
    side to show the peel-off in both directions.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 4.0))
    if K_star is None:
        K_star = M.find_K(temperature_C=temperature_C)

    for i, d in enumerate(sorted(offsets, reverse=True)):
        weight = 0.75 - 0.5 * i / max(len(offsets) - 1, 1)
        shade = plt.cm.Greys(0.3 + 0.45 * (1 - weight))
        for sign in (-1, +1):
            _, tr = M.propagated_shoot(K_star + sign * d, T=T,
                                       temperature_C=temperature_C)
            ax.plot(tr.t, tr.depolarisation, color=shade, lw=1.0)

    _, tr = M.propagated_shoot(K_star, T=T, temperature_C=temperature_C)
    ax.plot(tr.t, tr.depolarisation, color=INK, lw=2.0)
    j = int(np.argmax(tr.depolarisation))
    _label(ax, tr.t[j], tr.depolarisation[j],
           f"$K$ = {K_star:.3f} msec$^{{-1}}$", INK, dx=6, dy=2)
    ax.annotate("$V \\to +\\infty$\n($K$ too small)", xy=(0.02, 0.93),
                xycoords="axes fraction", fontsize=8, color=INK_2, va="top")
    ax.annotate("$V \\to -\\infty$\n($K$ too large)", xy=(0.02, 0.12),
                xycoords="axes fraction", fontsize=8, color=INK_2, va="top")
    ax.set_ylim(-80, 150)
    ax.set_xlim(0, min(T, 1.8))
    return _finish(ax, "Fig. 15 (p. 528): solving eqn (31) by shooting on $K$, "
                       f"{temperature_C} $^\\circ$C",
                   "msec", "$-V$ (mV)",
                   note="grey pairs bracket the critical value by "
                        f"{max(offsets):g} down to {min(offsets):g} msec$^{{-1}}$")


# --- Figs. 20, 21, 22: what the model predicts ----------------------------

def refractory_period(delays=(4, 6, 8, 12, 18), ax=None, first=15, second=90,
                      temperature_C=M.T_BASE):
    """Fig. 20, p. 532: a second shock at increasing delay after the first."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.8))
    shades = plt.cm.Greys(np.linspace(0.40, 0.95, len(delays)))
    for d, shade in zip(delays, shades):
        tr = M.membrane_action_potential(first, T=d + 14,
                                         shocks=[(d, second)],
                                         temperature_C=temperature_C)
        mask = tr.t >= d
        t, y = tr.t[mask] - d, tr.depolarisation[mask]
        # Five series: the peaks pile up, so identity goes in the legend.
        ax.plot(t, y, color=shade, lw=1.4, label=f"{d} ms")
    ax.set_xlim(0, 10)
    ax.set_ylim(-25, 125)
    ax.legend(loc="upper right", title="delay after first shock",
              title_fontsize=8, labelcolor=INK_2)
    return _finish(ax, "Fig. 20: the refractory period, from the equations alone",
                   "msec after the second shock", "$-V$ (mV)",
                   note=f"a {second} mV shock delivered at each delay after a "
                        f"{first} mV spike")


def find_threshold(lo=5.0, hi=8.0, iterations=30, **kwargs) -> float:
    """Bisect for the smallest shock that produces a spike, in mV."""
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if M.membrane_action_potential(mid, **kwargs).fired():
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def threshold(shocks=(5.5, 6.0, 6.45, 6.6, 7.5), ax=None, T=16.0,
              temperature_C=M.T_BASE):
    """Fig. 21, p. 534: the all-or-nothing boundary, resolved finely.

    H&H could only say the threshold lay between 6 and 7 mV, because each
    curve cost them days. Bisection puts it at 6.50 mV in a second.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 4.0))
    for s in shocks:
        tr = M.membrane_action_potential(s, T=T, temperature_C=temperature_C)
        fired = tr.fired()
        # The subthreshold curves all decay to the same place, so identity goes
        # in the legend; only the spikes get a direct label.
        ax.plot(tr.t, tr.depolarisation, color=INK if fired else MUTED,
                lw=1.7 if fired else 1.1,
                label=f"{s:g} mV" + ("" if fired else "  (no spike)"))
        if fired:
            j = int(np.argmax(tr.depolarisation))
            _label(ax, tr.t[j], tr.depolarisation[j], f"{s:g} mV", INK,
                   dx=4, dy=2, fontsize=7.5)
    ax.set_ylim(-25, 125)
    ax.legend(loc="center right", title="shock strength", title_fontsize=8,
              labelcolor=INK_2)
    thr = find_threshold(T=T, temperature_C=temperature_C)
    return _finish(ax, f"Fig. 21: threshold sits at {thr:.2f} mV",
                   "msec", "$-V$ (mV)",
                   note="the model is all-or-nothing without having been "
                        "told to be")


def anode_break(hold=30, ax=None, T=25.0, temperature_C=M.T_BASE):
    """Fig. 22A, p. 536: a spike produced by *ending* a hyperpolarisation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.8))
    tr = M.membrane_action_potential(hold_V=hold, T=T,
                                     temperature_C=temperature_C)
    ax.plot(tr.t, tr.depolarisation, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.annotate(f"held at $-V = -{hold}$ mV\nuntil $t = 0$", xy=(0.25, -hold),
                xytext=(0.30, 0.16), textcoords="axes fraction",
                fontsize=8, color=INK_2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                connectionstyle="arc3,rad=0.15"))
    ax.set_ylim(-45, 128)
    return _finish(ax, "Fig. 22: anode break excitation",
                   "msec", "$-V$ (mV)",
                   note="no stimulus is applied; the spike follows from "
                        "releasing the hyperpolarisation")


# --- A check H&H could not run --------------------------------------------

def integrator_comparison(depolarisation=15, axes=None, T=12.0,
                          steps=(0.05, 0.02, 0.01, 0.005, 0.002)):
    """Forward Euler at several step sizes against an adaptive solver.

    H&H used Hartree's method with a step "varied between about 0.01 msec at
    the beginning of a run ... and 1 msec during the small oscillations"
    (p. 523), and wrote that they were "confident that the overall errors are
    not large enough to be detected in the illustrations of this paper".

    Two errors are reported, because they say different things. The maximum
    pointwise error is dominated by the near-vertical rising phase, where a
    tiny shift in timing registers as a large difference in voltage. The error
    in peak height is what a reader of their figures would actually have seen.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10.0, 3.6),
                               constrained_layout=True)
    reference = M.membrane_action_potential(depolarisation, T=T, dt=0.001)
    pointwise, peak = [], []
    for dt in steps:
        tr = M.membrane_action_potential(depolarisation, T=T, dt=dt,
                                         method="euler")
        interp = np.interp(tr.t, reference.t, reference.depolarisation)
        pointwise.append(np.abs(tr.depolarisation - interp).max())
        peak.append(abs(tr.peak_depolarisation - reference.peak_depolarisation))

    panels = [
        (axes[0], pointwise, "max pointwise error in $-V$ (mV)",
         "Maximum pointwise error"),
        (axes[1], peak, "error in peak height (mV)",
         "Error in peak height"),
    ]
    for ax, values, ylabel, title in panels:
        ax.plot(steps, values, color=K, marker="o", markersize=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xticks(list(steps))
        ax.set_xticklabels([f"{v:g}" for v in steps])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.axvspan(0.02, 0.01, color="#f0efe9", zorder=0)
        for dt, v in zip(steps, values):
            _label(ax, dt, v, f"{v:.2g}", K, dx=0, dy=9, fontsize=7.5,
                   ha="center")
        _finish(ax, title, "step size (msec)", ylabel)
    axes[0].annotate("H&H's own step", xy=(0.0141, pointwise[1]),
                     xytext=(0, -26), textcoords="offset points",
                     fontsize=7.5, color=MUTED, ha="center")
    return axes, {"pointwise": dict(zip(steps, pointwise)),
                  "peak": dict(zip(steps, peak))}
