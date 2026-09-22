import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hh1952 import figures as F
F.use_paper_style()
jobs = [
    ("01_circuit", F.equivalent_circuit),
    ("02_rates", F.rate_constants),
    ("03_steady", F.steady_states),
    ("04_tau", F.time_constants),
    ("05_clamp", F.clamp_conductances),
    ("06_ap", F.membrane_action_potentials),
    ("07_ap_cond", F.action_potential_with_conductances),
    ("08_gates", F.gating_variables),
    ("09_shoot", F.propagated_shooting),
    ("10_refractory", F.refractory_period),
    ("11_threshold", F.threshold),
    ("12_anode", F.anode_break),
    ("13_converge", F.integrator_comparison),
]
for name, fn in jobs:
    try:
        r = fn()
        ax = r[0] if isinstance(r, tuple) else r
        fig = (ax if not hasattr(ax, "__len__") else ax.ravel()[0]).figure
        fig.savefig(f"figures/{name}.png")
        plt.close(fig)
        print("ok  ", name)
    except Exception as e:
        print("FAIL", name, type(e).__name__, e)
