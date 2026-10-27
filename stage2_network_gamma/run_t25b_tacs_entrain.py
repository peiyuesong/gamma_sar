"""T2.5b: tACS 40Hz 网络夹带初测（利用 I_GABA_B 通道注入正弦波）。

绘图 v3：
- rate 作为主曲线（反映真实活跃度，单调上升）
- f0 分段着色：subharmonic / transitional / entrained
- 4 个阶段区域 + 参考线，清晰展示"3+1 阶段"
- CSV 落盘，便于后续重画
"""
import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, run, defaultclock, seed, ms, mV, mS, cm,
                    uA, Hz, StateMonitor, SpikeMonitor, Network, Synapses,
                    TimedArray)
from common.networks import make_ing
from common.params import PARAMS

N = 100
I_MU = 0.85
SEED = 42
DT_MS = 0.025
F_TACS = 40.0
AMP_TACS_LIST = [0.0, 0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 5.0, 10.0]

T_DISCARD = 1000.0
T_MEASURE = 2000.0
T_TOTAL = T_DISCARD + T_MEASURE

results = []
for amp in AMP_TACS_LIST:
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

    t_arr = np.arange(0, T_TOTAL, DT_MS)
    I_wave = amp * np.sin(2 * np.pi * F_TACS * (t_arr / 1000.0))
    I_timed = TimedArray(I_wave * uA / (cm**2), dt=DT_MS * ms)

    tacs_syn = Synapses(
        G, G,
        model="""
        I_GABA_B_post = I_stim : amp/meter**2 (summed)
        I_stim = I_timed(t) : amp/meter**2
        """,
    )
    tacs_syn.connect(condition="i == j")
    tacs_syn.namespace['I_timed'] = I_timed

    sp = SpikeMonitor(G)
    net = Network(G, syn_ii, gap, tacs_syn, sp)
    net.run(T_TOTAL * ms)

    all_t = np.array(sp.t / ms)
    t = all_t[all_t >= T_DISCARD] - T_DISCARD
    rate = len(t) / (N * (T_MEASURE / 1000.0))

    bins = np.arange(0, T_MEASURE + 1, 1.0)
    hist, _ = np.histogram(t, bins=bins)
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, d=1e-3)
    band = (freqs >= 10) & (freqs <= 100)
    f0 = freqs[band][np.argmax(spec[band])] if spec[band].sum() > 0 else 0.0

    phase = (t / 1000.0 * F_TACS) % 1.0
    plv = np.abs(np.mean(np.exp(1j * 2 * np.pi * phase))) if len(phase) > 0 else 0.0

    results.append((amp, rate, f0, plv))
    print(f"amp={amp:5.2f} | rate={rate:5.1f} | f0={f0:5.1f} | PLV={plv:.2f}")

# CSV 落盘
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t25b_entrain.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["amp", "rate", "f0", "plv"])
    for row in results:
        w.writerow(row)
print(f"\nCSV saved: {csv_path}")

# ============ 绘图 ============
amps = np.array([r[0] for r in results])
rates = np.array([r[1] for r in results])
f0s = np.array([r[2] for r in results])

fig, ax = plt.subplots(figsize=(10, 6))

# 只画 amp <= 2.5 的细节区
mask = amps <= 2.5
amps_p, rates_p, f0s_p = amps[mask], rates[mask], f0s[mask]

# rate 主曲线（橙，单调上升）
ax.plot(amps_p, rates_p, 's-', color='#ff7f0e', lw=2.5, markersize=9,
        label='Firing rate (per cell)', zorder=2, alpha=0.85)

# f0 分段
sub_mask = f0s_p < 25
ent_mask = (f0s_p >= 38) & (f0s_p <= 42)
trans_mask = ~sub_mask & ~ent_mask

ax.plot(amps_p[sub_mask], f0s_p[sub_mask], 'o-', color='#1f77b4', lw=2.5,
        markersize=11, label='f0: subharmonic (20Hz)', zorder=3)
ax.plot(amps_p[ent_mask], f0s_p[ent_mask], 'o-', color='#2ca02c', lw=2.5,
        markersize=11, label='f0: entrained (40Hz)', zorder=3)
ax.scatter(amps_p[trans_mask], f0s_p[trans_mask], s=180, c='#d62728', marker='X',
           zorder=4, label='f0: transitional (mixed PSD)', edgecolors='black')

# 参考线
ax.axhline(40, color='green', ls='--', lw=1.2, alpha=0.5)
ax.axhline(20, color='blue', ls='--', lw=1.2, alpha=0.5)
ax.axhline(36.5, color='purple', ls=':', lw=1.2, alpha=0.5)

# 四阶段区域
ax.axvspan(-0.15, 0.25, color='lightgray', alpha=0.2)
ax.axvspan(0.3, 1.05, color='lightblue', alpha=0.15)
ax.axvspan(1.1, 1.65, color='mistyrose', alpha=0.35)
ax.axvspan(1.75, 2.55, color='honeydew', alpha=0.4)

# 区域标注
for x, y, txt, col in [
    (0.05, 68, 'Uncoupled', 'dimgray'),
    (0.65, 68, 'Subharmonic lock', 'steelblue'),
    (1.35, 68, 'Transition', 'darkred'),
    (2.15, 68, 'Full 1:1 lock', 'darkgreen'),
]:
    ax.text(x, y, txt, fontsize=10, color=col, ha='center',
            bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='none'))

ax.set_xlabel("tACS amplitude (μA/cm²)", fontsize=12)
ax.set_ylabel("Frequency (Hz)", fontsize=12)
ax.set_title("T2.5b: tACS 40Hz entrainment dose-response (N=100)", fontsize=13)
ax.set_xlim(-0.15, 2.55)
ax.set_ylim(15, 75)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "figures", "f_t25b_tacs_entrainment.png")
plt.savefig(fig_path, dpi=150)
print(f"saved: {fig_path}")