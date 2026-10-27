"""T2.6d: 10-seed statistical comparison (EO vs EC × Sham/iTBS/tACS+iTBS)."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, run, defaultclock, seed, ms, uA, cm,
                    SpikeMonitor, Network, Synapses, TimedArray)
from common.networks import make_ing
from common.params import PARAMS

N = 100
I_MU = 0.80
DT_MS = 0.025

T_BASE = 1000.0
T_STIM = 2000.0
T_POST = 1000.0
T_TOTAL = T_BASE + T_STIM + T_POST
N_STEPS = int(T_TOTAL / DT_MS) + 1

G_KV3_LIST = [40.0, 20.0]
STIM_MODES = ["Sham", "iTBS", "tACS+iTBS"]
SEED_LIST = list(range(10))
AMP_TACS = 2.0
F_TACS = 40.0

pg = PARAMS["gabab"]
n_coop = pg.n_coop
K1_g = pg.K1 * 1000.0
K2_g = pg.K2 * 1000.0
K3_eff_g = (pg.K3 / (pg.K_d ** (1.0 / n_coop))) * 1000.0
K4_g = pg.K4 * 1000.0

def run_sim(g_kv3, stim_mode, seed_val, g_gabab_val=0.005):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(seed_val)

    cell = PARAMS["cell"]
    saved = (cell.g_Kv3, cell.g_Kdr)
    cell.g_Kv3, cell.g_Kdr = g_kv3, 20.0
    try:
        G, syn_ii, gap = make_ing(N, i_mu=I_MU, seed=seed_val)
    finally:
        cell.g_Kv3, cell.g_Kdr = saved

    objects = [G, syn_ii, gap]

    t_arr = np.arange(N_STEPS) * DT_MS
    I_wave = np.zeros(N_STEPS)

    if "tACS" in stim_mode:
        mask = (t_arr >= T_BASE) & (t_arr < T_BASE + T_STIM)
        I_wave[mask] += AMP_TACS * np.sin(2 * np.pi * F_TACS * t_arr[mask] / 1000.0)

    if "iTBS" in stim_mode:
        dt_fine = 0.001
        n_fine = int(T_TOTAL / dt_fine) + 1
        r_traj = np.zeros(n_fine)
        s_traj = np.zeros(n_fine)
        t_fine = np.arange(n_fine) * dt_fine
        for i in range(1, n_fine):
            t_ms = i * dt_fine
            T_val = 0.0
            if T_BASE <= t_ms < T_BASE + T_STIM:
                if (t_ms - T_BASE) % 100.0 < 20.0:
                    T_val = 1.0
            dr = (K1_g * T_val * (1 - r_traj[i-1]) - K2_g * r_traj[i-1]) * (dt_fine / 1000.0)
            ds = (K3_eff_g * r_traj[i-1] - K4_g * s_traj[i-1]) * (dt_fine / 1000.0)
            r_traj[i] = max(0.0, min(1.0, r_traj[i-1] + dr))
            s_traj[i] = max(0.0, min(1.0, s_traj[i-1] + ds))
        s_coarse = np.interp(t_arr, t_fine, s_traj)
        g_factor = s_coarse**n_coop / (s_coarse**n_coop + 1.0)
        I_wave += -g_gabab_val * g_factor * 25.0

    I_timed = TimedArray(I_wave * uA / (cm**2), dt=DT_MS * ms)
    stim_syn = Synapses(G, G, model="""
        I_GABA_B_post = I_stim : amp/meter**2 (summed)
        I_stim = I_timed(t) : amp/meter**2
    """)
    stim_syn.connect(condition="i == j")
    stim_syn.namespace['I_timed'] = I_timed
    objects.append(stim_syn)

    sp = SpikeMonitor(G)
    objects.append(sp)
    net = Network(*objects)
    net.run(T_TOTAL * ms)

    all_t = np.array(sp.t / ms)
    t_stim = all_t[(all_t >= T_BASE + 500) & (all_t < T_BASE + T_STIM)] - (T_BASE + 500)

    if len(t_stim) < 10: return 0.0
    bins = np.arange(0, T_STIM + 1, 1.0)
    hist, _ = np.histogram(t_stim, bins=bins)
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, d=1e-3)
    band = (freqs >= 30) & (freqs <= 50)
    return spec[band].sum()

print("=" * 60)
print("T2.6d: 10-seed statistical comparison")
print("=" * 60)

data = {}
for g_kv3 in G_KV3_LIST:
    state = "EO" if g_kv3 == 40.0 else "EC"
    for mode in STIM_MODES:
        vals = []
        for sd in SEED_LIST:
            power = run_sim(g_kv3, mode, sd)
            vals.append(power)
            print(f"  seed={sd:2d} power={power:.1f}")
        data[(state, mode)] = np.array(vals)
        print(f"{state} | {mode:10s} | mean={np.mean(vals):.1f} ± {np.std(vals):.1f}")

# --- raw per-seed CSV (so every number in Fig 4a is traceable) ---
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t26d_stats.csv")
with open(csv_path, "w", encoding="utf-8") as fh:
    fh.write("i_mu,g_kv3,state,mode,seed,gamma_power_30_50\n")
    for g_kv3 in G_KV3_LIST:
        state = "EO" if g_kv3 == 40.0 else "EC"
        for mode in STIM_MODES:
            for sd, val in zip(SEED_LIST, data[(state, mode)]):
                fh.write(f"{I_MU},{g_kv3},{state},{mode},{sd},{val:.6f}\n")
print(f"\nsaved CSV: {csv_path}")

print("\n=== Wilcoxon signed-rank test (vs Sham) ===")
for state in ["EO", "EC"]:
    sham = data[(state, "Sham")]
    for mode in ["iTBS", "tACS+iTBS"]:
        vals = data[(state, mode)]
        if np.all(vals == sham):
            print(f"{state} | {mode:10s} vs Sham | no difference (p=1.0)")
            continue
        try:
            stat, p = wilcoxon(vals, sham)
            print(f"{state} | {mode:10s} vs Sham | p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")
        except ValueError:
            print(f"{state} | {mode:10s} vs Sham | unable to compute")

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(STIM_MODES))
width = 0.35

for i, state in enumerate(["EO", "EC"]):
    means = [np.mean(data[(state, m)]) for m in STIM_MODES]
    stds = [np.std(data[(state, m)]) for m in STIM_MODES]
    ax.bar(x + (i - 0.5)*width, means, width, yerr=stds, capsize=5, label=state)

ax.set_xticks(x)
ax.set_xticklabels(STIM_MODES)
ax.set_ylabel("Stimulus-period gamma power")
ax.set_title("T2.6d: 10-seed statistical comparison")
ax.legend()
plt.tight_layout()

fig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "f_t26d_stats.png")
plt.savefig(fig_path, dpi=150)
print(f"\nsaved: {fig_path}")