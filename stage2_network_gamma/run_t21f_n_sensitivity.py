"""T2.1f: N 敏感性扫描（N=50/100/200 × seeds=1/7/42）。

验证 ING 网络的 f0 和同步度是否对网络规模敏感，
为后续外推和审稿人质疑提供定量护身符。

模型来源：复用 common.networks.make_ing（与 T2.1d/T2.5/T2.6/T2.7 同源），
保证 N 敏感性与 Stage 2 其他数据可比。
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, run, defaultclock, seed, ms,
                    SpikeMonitor, Network)
from common.networks import make_ing
from common.params import PARAMS

I_MU = 0.85
SEED_LIST = [1, 7, 42]
N_LIST = [50, 100, 200]
DT_MS = 0.025
T_DISCARD = 1000.0
T_MEASURE = 2000.0
T_TOTAL = T_DISCARD + T_MEASURE

results = []

for N in N_LIST:
    for sd in SEED_LIST:
        start_scope()
        defaultclock.dt = DT_MS * ms
        seed(sd)

        cell = PARAMS["cell"]
        saved = (cell.g_Kv3, cell.g_Kdr)
        cell.g_Kv3, cell.g_Kdr = 40.0, 20.0
        try:
            G, syn_ii, gap = make_ing(N, i_mu=I_MU, seed=sd)
        finally:
            cell.g_Kv3, cell.g_Kdr = saved

        sp = SpikeMonitor(G)
        net = Network(G, syn_ii, gap, sp)
        net.run(T_TOTAL * ms)

        all_t = np.array(sp.t / ms)
        all_i = np.array(sp.i, dtype=int)
        t = all_t[all_t >= T_DISCARD] - T_DISCARD
        idx = all_i[all_t >= T_DISCARD]

        rate = len(t) / (N * (T_MEASURE / 1000.0))

        bins = np.arange(0, T_MEASURE + 1, 1.0)
        hist, _ = np.histogram(t, bins=bins)
        r = hist - hist.mean()
        spec = np.abs(np.fft.rfft(r, 8192)) ** 2
        freqs = np.fft.rfftfreq(8192, d=1e-3)
        band = (freqs >= 10) & (freqs <= 100)
        f0 = freqs[band][np.argmax(spec[band])] if spec[band].sum() > 0 else 0.0

        if len(t) > N:
            M, _, _ = np.histogram2d(idx, t, bins=[np.arange(N + 1), bins])
            Mc = M - M.mean(axis=1, keepdims=True)
            valid = Mc.std(axis=1) > 0
            if valid.sum() > 1:
                Mv = Mc[valid]
                Mv = Mv / (Mv.std(axis=1, keepdims=True) + 1e-12)
                corr_mat = (Mv @ Mv.T) / Mv.shape[1]
                nv = corr_mat.shape[0]
                sync_r = (corr_mat.sum() - np.trace(corr_mat)) / (nv * (nv - 1))
            else:
                sync_r = 0.0
        else:
            sync_r = 0.0

        results.append((N, sd, rate, f0, sync_r))
        print(f"N={N:3d} seed={sd:2d} | rate={rate:5.1f} | f0={f0:5.1f} | sync_r={sync_r:.3f}")

print("\n" + "=" * 50)
print("N 敏感性汇总")
print("=" * 50)
N_arr = np.array([r[0] for r in results])
f0_arr = np.array([r[3] for r in results])
rate_arr = np.array([r[2] for r in results])
sync_arr = np.array([r[4] for r in results])

for N in N_LIST:
    mask = N_arr == N
    print(f"  N={N:3d}: f0 = {f0_arr[mask].mean():5.1f} ± {f0_arr[mask].std():.1f} Hz | "
          f"rate = {rate_arr[mask].mean():5.1f} Hz | sync_r = {sync_arr[mask].mean():.3f}")

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t21f_n_sensitivity.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["N", "seed", "rate_hz", "f0_hz", "sync_r"])
    for row in results:
        w.writerow(row)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax1, ax2 = axes

f0_means = [f0_arr[N_arr == N].mean() for N in N_LIST]
f0_stds = [f0_arr[N_arr == N].std() for N in N_LIST]
ax1.errorbar(N_LIST, f0_means, yerr=f0_stds, fmt='o-', capsize=5,
             color='#1f77b4', lw=2, markersize=8)
ax1.axhline(40, color='red', ls='--', alpha=0.5, label='tACS 40Hz')
ax1.set_xlabel("Network size N")
ax1.set_ylabel("f0 (Hz)")
ax1.set_title("f0 vs Network Size")
ax1.set_xscale('log')
ax1.set_xticks(N_LIST)
ax1.set_xticklabels([str(n) for n in N_LIST])
ax1.legend()

sync_means = [sync_arr[N_arr == N].mean() for N in N_LIST]
sync_stds = [sync_arr[N_arr == N].std() for N in N_LIST]
ax2.errorbar(N_LIST, sync_means, yerr=sync_stds, fmt='s-', capsize=5,
             color='#2ca02c', lw=2, markersize=8)
ax2.set_xlabel("Network size N")
ax2.set_ylabel("Pairwise correlation (sync_r)")
ax2.set_title("Synchrony vs Network Size")
ax2.set_xscale('log')
ax2.set_xticks(N_LIST)
ax2.set_xticklabels([str(n) for n in N_LIST])

plt.tight_layout()
fig_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "f_t21f_n_sensitivity.png")
plt.savefig(fig_path, dpi=150)
print(f"\nsaved: {fig_path}")