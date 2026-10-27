"""T2.4b: GABA_B 微环路验证 v7。

复用 make_ing，注入 GABA_B 电流。
修复：目标变量改为专用 I_GABA_B，避免与 I_syn 求和冲突。
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, run, defaultclock, seed, ms, mV, mS, cm,
                    uA, mM, Hz, StateMonitor, SpikeMonitor, Network,
                    Synapses, TimedArray)
from common.networks import make_ing
from common.params import PARAMS

pg = PARAMS["gabab"]

N = 100
I_MU = 0.85
SEED = 42
DT_MS = 0.025
g_GABA_B_val = 0.05

T_TOTAL = 3500.0
T_ON = 500.0
T_OFF = 2500.0
N_STEPS = int(T_TOTAL / DT_MS) + 1

T_arr = np.zeros(N_STEPS)
T_arr[int(T_ON / DT_MS):int(T_OFF / DT_MS)] = 1.0
T_timed = TimedArray(T_arr * mM, dt=DT_MS * ms)

start_scope()
defaultclock.dt = DT_MS * ms
seed(SEED)

cell = PARAMS["cell"]
saved = (cell.g_Kv3, cell.g_Kdr)
cell.g_Kv3, cell.g_Kdr = 40.0, 20.0
try:
    G, syn_ii, gap = make_ing(N, i_mu=I_MU, seed=SEED)
finally:
    cell.g_Kv3, cell.g_Kdr = saved

# 突触模型：目标变量改为 I_GABA_B_post
gaba_b_syn = Synapses(
    G, G,
    model="""
    dr_g/dt = (K1_g * T_g * (1 - r_g) - K2_g * r_g) : 1 (clock-driven)
    ds_g/dt = (K3_eff_g * r_g - K4_g * s_g) : 1 (clock-driven)
    T_g = T_timed(t) : mM

    I_GABA_B_post = g_GABA_B_g * (s_g**n_coop_g / (s_g**n_coop_g + 1)) * (E_K_g - v_post) : amp/meter**2 (summed)

    g_GABA_B_g : siemens/meter**2
    K1_g : Hz/mM
    K2_g : Hz
    K3_eff_g : Hz
    K4_g : Hz
    n_coop_g : 1
    E_K_g : volt
    """
)
gaba_b_syn.connect(condition="i == j")

n_coop = pg.n_coop
gaba_b_syn.g_GABA_B_g = g_GABA_B_val * mS / (cm**2)
gaba_b_syn.K1_g = pg.K1 * 1000.0 * Hz / mM
gaba_b_syn.K2_g = pg.K2 * 1000.0 * Hz
gaba_b_syn.K3_eff_g = (pg.K3 / (pg.K_d ** (1.0 / n_coop))) * 1000.0 * Hz
gaba_b_syn.K4_g = pg.K4 * 1000.0 * Hz
gaba_b_syn.n_coop_g = n_coop
gaba_b_syn.E_K_g = -90.0 * mV

gaba_b_syn.r_g = 0.0
gaba_b_syn.s_g = 0.0

sp = SpikeMonitor(G)
mon_G = StateMonitor(G, "v", record=[0])
mon_S = StateMonitor(gaba_b_syn, ["s_g", "r_g"], record=[0])

net = Network(G, syn_ii, gap, gaba_b_syn, sp, mon_G, mon_S)
net.run(T_TOTAL * ms)

t_arr = np.array(mon_G.t / ms)
v_arr = np.array(mon_G.v[0] / mV)
s_arr = np.array(mon_S.s_g[0])

all_t = np.array(sp.t / ms)

window = 50.0
bins = np.arange(0, T_TOTAL + window, window)
rates = np.zeros(len(bins) - 1)
for j in range(len(bins) - 1):
    rates[j] = np.sum((all_t >= bins[j]) & (all_t < bins[j+1])) / (N * window / 1000.0)
bin_centers = bins[:-1] + window / 2.0

baseline_mask = (bin_centers >= 100) & (bin_centers < T_ON)
active_mask = (bin_centers >= T_ON + 500) & (bin_centers < T_OFF)

baseline_rate = np.mean(rates[baseline_mask]) if baseline_mask.any() else 0
active_rate = np.mean(rates[active_mask]) if active_mask.any() else 0
drop_pct = ((baseline_rate - active_rate) / baseline_rate * 100) if baseline_rate > 0 else 0

print("=" * 50)
print("T2.4b GABA_B 微环路验证 v7")
print("=" * 50)
print(f"  N={N}, i_mu={I_MU}, g_GABA_B={g_GABA_B_val}")
print(f"  基线放电率 = {baseline_rate:.1f} Hz")
print(f"  激活期放电率 = {active_rate:.1f} Hz")
print(f"  下降幅度 = {drop_pct:.1f}%  (>= 30% ?)")
verdict = "PASS" if drop_pct >= 30 else "ADJUST" if drop_pct > 0 else "FAIL"
print(f"  verdict = {verdict}")
print("=" * 50)

fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
ax1, ax2, ax3 = axes

ax1.plot(t_arr, v_arr, 'b-', lw=0.5)
ax1.axvspan(T_ON, T_OFF, color='r', alpha=0.1)
ax1.set_ylabel("V (mV)")
ax1.set_title("representative seed")

ax2.plot(t_arr, s_arr, 'g-', lw=1.0)
ax2.axvspan(T_ON, T_OFF, color='r', alpha=0.1)
ax2.set_ylabel("s_g (norm)")

ax3.plot(bin_centers, rates, 'k-', lw=1.5)
ax3.axvspan(T_ON, T_OFF, color='r', alpha=0.1)
ax3.axhline(baseline_rate, color='gray', ls='--', alpha=0.5)
ax3.set_ylabel("Rate (Hz)")
ax3.set_xlabel("Time (ms)")

plt.tight_layout()
fig_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "f_t24b_gabab_network.png"
)
plt.savefig(fig_path, dpi=150)
print(f"saved: {fig_path}")