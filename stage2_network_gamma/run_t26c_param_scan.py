"""T2.6c: EO/EC 分叉参数扫描.

固定 i_mu=0.80，扫描 GABA_B × tACS 的二维参数空间，
找"EO 能救、EC 不能救"的临界窗口。

网格：
  GABA_B: 0.001, 0.002, 0.003, 0.005 (mS/cm2)
  tACS:   0.0,   1.0,   1.5,   2.0   (uA/cm2)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, run, defaultclock, seed, ms, mV, mS, cm,
                    uA, mM, Hz, SpikeMonitor, Network, Synapses, TimedArray)
from common.networks import make_ing
from common.params import PARAMS

N = 100
I_MU = 0.80
SEED = 42
DT_MS = 0.025

T_BASE = 1000.0
T_STIM = 2000.0
T_POST = 1000.0
T_TOTAL = T_BASE + T_STIM + T_POST
N_STEPS = int(T_TOTAL / DT_MS) + 1

G_KV3_LIST = [40.0, 20.0]
GABA_B_LIST = [0.001, 0.002, 0.003, 0.005]
TACS_LIST = [0.0, 1.0, 1.5, 2.0]
F_TACS = 40.0

pg = PARAMS["gabab"]
n_coop = pg.n_coop
K1_g = pg.K1 * 1000.0
K2_g = pg.K2 * 1000.0
K3_eff_g = (pg.K3 / (pg.K_d ** (1.0 / n_coop))) * 1000.0
K4_g = pg.K4 * 1000.0

# 预计算 iTBS 包络
t_arr = np.arange(N_STEPS) * DT_MS
T_gaba_arr = np.zeros(N_STEPS)
mask = (t_arr >= T_BASE) & (t_arr < T_BASE + T_STIM)
t_rel = (t_arr[mask] - T_BASE) % 100.0
T_gaba_arr[mask] = np.where(t_rel < 20.0, 1.0, 0.0)

def make_gaba_b_traj(g_gabab_val):
    dt_fine = 0.001
    n_fine = int(T_TOTAL / dt_fine) + 1
    r_traj = np.zeros(n_fine)
    s_traj = np.zeros(n_fine)
    for i in range(1, n_fine):
        t_ms = i * dt_fine
        T_input = T_gaba_arr[min(int(t_ms / DT_MS), N_STEPS - 1)]
        dr = (K1_g * T_input * (1 - r_traj[i-1]) - K2_g * r_traj[i-1]) * (dt_fine / 1000.0)
        ds = (K3_eff_g * r_traj[i-1] - K4_g * s_traj[i-1]) * (dt_fine / 1000.0)
        r_traj[i] = max(0.0, min(1.0, r_traj[i-1] + dr))
        s_traj[i] = max(0.0, s_traj[i-1] + ds)
    t_fine = np.arange(n_fine) * dt_fine
    t_coarse = np.arange(N_STEPS) * DT_MS
    s_coarse = np.interp(t_coarse, t_fine, s_traj)
    g_factor = s_coarse**n_coop / (s_coarse**n_coop + 1.0)
    return -g_gabab_val * g_factor * 25.0

def run_sim(g_kv3, g_gabab_val, amp_tacs):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(SEED)

    cell = PARAMS["cell"]
    saved = (cell.g_Kv3, cell.g_Kdr)
    cell.g_Kv3, cell.g_Kdr = g_kv3, 20.0
    try:
        G, syn_ii, gap = make_ing(N, i_mu=I_MU, seed=SEED)
    finally:
        cell.g_Kv3, cell.g_Kdr = saved

    I_tacs_arr = np.zeros(N_STEPS)
    if amp_tacs > 0:
        mask = (t_arr >= T_BASE) & (t_arr < T_BASE + T_STIM)
        I_tacs_arr[mask] = amp_tacs * np.sin(2 * np.pi * F_TACS * t_arr[mask] / 1000.0)
    I_gaba_arr = make_gaba_b_traj(g_gabab_val) if g_gabab_val > 0 else np.zeros(N_STEPS)
    I_wave = I_tacs_arr + I_gaba_arr
    I_timed = TimedArray(I_wave * uA / (cm**2), dt=DT_MS * ms)

    stim_syn = Synapses(
        G, G,
        model="""
        I_GABA_B_post = I_stim : amp/meter**2 (summed)
        I_stim = I_timed(t) : amp/meter**2
        """,
    )
    stim_syn.connect(condition="i == j")
    stim_syn.namespace['I_timed'] = I_timed

    sp = SpikeMonitor(G)
    net = Network(G, syn_ii, gap, stim_syn, sp)
    net.run(T_TOTAL * ms)

    all_t = np.array(sp.t / ms)
    t_stim = all_t[(all_t >= T_BASE + 500) & (all_t < T_BASE + T_STIM)] - (T_BASE + 500)

    rate = len(t_stim) / (N * (T_STIM - 500)/1000.0) if len(t_stim) > 0 else 0

    if len(t_stim) > 0:
        bins = np.arange(0, T_STIM + 1, 1.0)
        hist, _ = np.histogram(t_stim, bins=bins)
        spec = np.abs(np.fft.rfft(hist - hist.mean(), 8192)) ** 2
        freqs = np.fft.rfftfreq(8192, d=1e-3)
        band = (freqs >= 10) & (freqs <= 100)
        f0 = freqs[band][np.argmax(spec[band])] if spec[band].sum() > 0 else 0.0
    else:
        f0 = 0.0

    return rate, f0

# ============ 扫描 ============
print("=" * 75)
print("T2.6c 参数扫描: i_mu=0.80, 找 EO/EC 分叉窗口")
print("=" * 75)

results = {}
for g_gabab in GABA_B_LIST:
    for amp_tacs in TACS_LIST:
        print(f"\nGABA_B={g_gabab:.3f}, tACS={amp_tacs:.1f}:")
        for g_kv3 in G_KV3_LIST:
            state = "EO" if g_kv3 == 40.0 else "EC"
            rate, f0 = run_sim(g_kv3, g_gabab, amp_tacs)
            results[(g_gabab, amp_tacs, g_kv3)] = (rate, f0)
            print(f"  {state}: rate={rate:5.1f}Hz  f0={f0:5.1f}Hz")

# ============ 汇总：找分叉点 ============
print("\n" + "=" * 75)
print("分叉分析 (EO rate - EC rate)")
print("=" * 75)
best_diff = 0
best_params = None
for g_gabab in GABA_B_LIST:
    for amp_tacs in TACS_LIST:
        eo_rate, eo_f0 = results[(g_gabab, amp_tacs, 40.0)]
        ec_rate, ec_f0 = results[(g_gabab, amp_tacs, 20.0)]
        diff = eo_rate - ec_rate
        eo_ok = abs(eo_f0 - 40.0) < 3.0 and eo_rate > 15
        ec_fail = ec_rate < 5 or abs(ec_f0 - 40.0) >= 5
        flag = "*** REVERSAL ***" if (eo_ok and ec_fail and diff > 5) else ""
        print(f"  gB={g_gabab:.3f} tACS={amp_tacs:.1f}: EO({eo_rate:4.1f},{eo_f0:4.1f}) "
              f"EC({ec_rate:4.1f},{ec_f0:4.1f}) diff={diff:5.1f} {flag}")
        if diff > best_diff:
            best_diff = diff
            best_params = (g_gabab, amp_tacs)

if best_params:
    print(f"\n  最大分叉: gB={best_params[0]:.3f}, tACS={best_params[1]:.1f}, diff={best_diff:.1f}Hz")
else:
    print("\n  本次扫描内未出现显著分叉")

# ============ 热力图 ============
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# EO rate
eo_rates = np.zeros((len(GABA_B_LIST), len(TACS_LIST)))
ec_rates = np.zeros((len(GABA_B_LIST), len(TACS_LIST)))
for i, g in enumerate(GABA_B_LIST):
    for j, a in enumerate(TACS_LIST):
        eo_rates[i, j] = results[(g, a, 40.0)][0]
        ec_rates[i, j] = results[(g, a, 20.0)][0]

im0 = axes[0].imshow(eo_rates, aspect='auto', cmap='RdYlGn', vmin=0, vmax=40)
axes[0].set_xticks(range(len(TACS_LIST)))
axes[0].set_xticklabels([f"{x:.1f}" for x in TACS_LIST])
axes[0].set_yticks(range(len(GABA_B_LIST)))
axes[0].set_yticklabels([f"{x:.3f}" for x in GABA_B_LIST])
axes[0].set_xlabel("tACS (uA/cm2)")
axes[0].set_ylabel("GABA_B (mS/cm2)")
axes[0].set_title("EO (gKv3=40) rate")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(ec_rates, aspect='auto', cmap='RdYlGn', vmin=0, vmax=40)
axes[1].set_xticks(range(len(TACS_LIST)))
axes[1].set_xticklabels([f"{x:.1f}" for x in TACS_LIST])
axes[1].set_yticks(range(len(GABA_B_LIST)))
axes[1].set_yticklabels([f"{x:.3f}" for x in GABA_B_LIST])
axes[1].set_xlabel("tACS (uA/cm2)")
axes[1].set_ylabel("GABA_B (mS/cm2)")
axes[1].set_title("EC (gKv3=20) rate")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
fig_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "f_t26c_param_scan.png"
)
plt.savefig(fig_path, dpi=150)
print(f"\nsaved: {fig_path}")