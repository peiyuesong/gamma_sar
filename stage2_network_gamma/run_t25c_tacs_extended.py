"""T2.5c: tACS 扩展扫描 — 亚阈值精细区 + 超阈值破坏区.

复用 make_ing，通过 I_GABA_B 通道注入 40Hz 正弦波。
覆盖 0.05–10.0 uA/cm2，探求真实的夹带阈值和超阈值行为。
"""
import os
import sys

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
AMP_TACS_LIST = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75,
                 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

T_DISCARD = 1000.0
T_MEASURE = 2000.0
T_TOTAL = T_DISCARD + T_MEASURE
N_STEPS = int(T_TOTAL / DT_MS) + 1

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

    # 构建 40Hz 正弦波
    t_arr = np.arange(0, T_TOTAL, DT_MS)
    I_wave = amp * np.sin(2 * np.pi * F_TACS * (t_arr / 1000.0))
    I_timed = TimedArray(I_wave * uA / (cm**2), dt=DT_MS * ms)

    # 用 Synapses 注入正弦电流到 I_GABA_B 通道（i==j）
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

    # 功率谱
    bins = np.arange(0, T_MEASURE + 1, 1.0)
    hist, _ = np.histogram(t, bins=bins)
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, d=1e-3)
    band = (freqs >= 10) & (freqs <= 100)
    f0 = freqs[band][np.argmax(spec[band])] if spec[band].sum() > 0 else 0.0

    stim_idx = np.argmin(np.abs(freqs - F_TACS))
    stim_power = spec[stim_idx]
    f0_power = spec[band][np.argmax(spec[band])]
    capture_ratio = stim_power / (stim_power + f0_power) if (stim_power + f0_power) > 0 else 0

    # PLV
    phase = (t / 1000.0 * F_TACS) % 1.0
    plv = np.abs(np.mean(np.exp(1j * 2 * np.pi * phase))) if len(phase) > 0 else 0.0

    results.append((amp, rate, f0, capture_ratio, plv))
    print(f"amp={amp:5.2f} | rate={rate:5.1f} | f0={f0:5.1f} | cap={capture_ratio:.2f} | PLV={plv:.2f}")

# 汇总
print("\n" + "=" * 70)
print("T2.5c tACS 扩展扫描汇总")
print("=" * 70)
for amp, rate, f0, cap, plv in results:
    entrained = "YES" if abs(f0 - F_TACS) < 3.0 else "PARTIAL" if cap > 0.3 else "NO"
    print(f"  amp={amp:5.2f}: rate={rate:5.1f} f0={f0:5.1f} [{entrained:7s}] cap={cap:.2f} PLV={plv:.2f}")

# 画图
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
amps = [r[0] for r in results]
rates = [r[1] for r in results]
f0s = [r[2] for r in results]
caps = [r[3] for r in results]
plvs = [r[4] for r in results]

ax1, ax2 = axes
ax1.semilogx(amps, f0s, 'o-', color='#1f77b4', lw=2, markersize=6, label='f0')
ax1.semilogx(amps, rates, 's--', color='#ff7f0e', lw=1.5, markersize=5, label='rate', alpha=0.7)
ax1.axhline(F_TACS, color='r', ls='--', alpha=0.4, label='tACS 40Hz')
ax1.axhline(20, color='gray', ls=':', alpha=0.4, label='1:1 mode 20Hz')
ax1.axhline(36.5, color='green', ls=':', alpha=0.4, label='base 2:1 36.5Hz')
ax1.set_ylabel("Frequency (Hz)")
ax1.axvspan(0.8, 1.5, color='yellow', alpha=0.15, label='chaotic transition')
ax1.set_title("T2.5c: tACS extended scan (sub- & supra-threshold)")
ax1.legend(loc='upper left', fontsize=8)
ax1.set_ylim(0, 80)

ax2.semilogx(amps, caps, 'D-', color='#2ca02c', lw=2, markersize=5, label='capture ratio')
ax2.semilogx(amps, plvs, '^--', color='#d62728', lw=1.5, markersize=5, label='PLV', alpha=0.7)
ax2.set_xlabel("tACS amplitude (uA/cm2)")
ax2.set_ylabel("Sync index")
ax2.set_ylim(0, 1.05)
ax2.legend(loc='upper left', fontsize=8)

plt.tight_layout()
fig_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "f_t25c_tacs_extended.png"
)
plt.savefig(fig_path, dpi=150)
print(f"\nsaved: {fig_path}")