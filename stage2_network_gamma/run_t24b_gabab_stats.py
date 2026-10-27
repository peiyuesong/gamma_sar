"""T2.4b: GABA_B network silencing — 10-seed statistical validation.

复用 make_ing 和原生 Python 解释器（绕开 cpp_standalone）。
验证：10 个不同随机种子下，2s 的 GABA_B 脉冲能否均导致 100% 静默。
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, run, defaultclock, seed, ms, uA, cm,
                    SpikeMonitor, Network, Synapses, TimedArray)
from common.networks import make_ing
from common.params import PARAMS

N = 100
I_MU = 0.80
DT_MS = 0.025
SEEDS = list(range(1, 11))

T_WARMUP = 500.0
T_STIM = 2000.0
T_POST = 1000.0
T_TOTAL = T_WARMUP + T_STIM + T_POST
N_STEPS = int(T_TOTAL / DT_MS) + 1

GABA_B_AMP = 0.005  # 与 T2.6d 保持一致，足以导致完全静默

results = []

for sd in SEEDS:
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

    objects = [G, syn_ii, gap]

    # 生成 GABA_B 电流波形（2s 恒定超极化）
    t_arr = np.arange(N_STEPS) * DT_MS
    I_wave = np.zeros(N_STEPS)
    mask = (t_arr >= T_WARMUP) & (t_arr < T_WARMUP + T_STIM)
    I_wave[mask] = -GABA_B_AMP * 25.0  # 等效超极化电流

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

    # 分析区间
    t_base = all_t[(all_t >= 100) & (all_t < T_WARMUP)]
    t_gaba = all_t[(all_t >= T_WARMUP + 200) & (all_t < T_WARMUP + T_STIM)]  # 给 200ms 反应时间
    t_post = all_t[(all_t >= T_WARMUP + T_STIM + 100) & (all_t < T_TOTAL)]

    r0 = len(t_base) / (N * (T_WARMUP - 100)/1000.0)
    r1 = len(t_gaba) / (N * (T_STIM - 200)/1000.0)
    r2 = len(t_post) / (N * (T_POST - 100)/1000.0)

    results.append((sd, r0, r1, r2))
    print(f"seed={sd:2d} | baseline={r0:5.1f} Hz | GABA_B={r1:5.1f} Hz | post={r2:5.1f} Hz")

# 统计
res_arr = np.array(results)
r0_all, r1_all, r2_all = res_arr[:, 1], res_arr[:, 2], res_arr[:, 3]

t_stat, p_val = stats.ttest_rel(r0_all, r1_all)
cohen_d = (r0_all.mean() - r1_all.mean()) / np.sqrt((np.var(r0_all, ddof=1) + np.var(r1_all, ddof=1))/2)

print("\n" + "=" * 50)
print("T2.4b Statistical Summary (n=10 seeds)")
print("=" * 50)
print(f"Baseline:   {r0_all.mean():.2f} ± {r0_all.std():.2f} Hz")
print(f"GABA_B:     {r1_all.mean():.2f} ± {r1_all.std():.2f} Hz")
print(f"Post:       {r2_all.mean():.2f} ± {r2_all.std():.2f} Hz")
print(f"Silencing:  {(1 - r1_all.mean() / r0_all.mean()) * 100:.1f}%")
print(f"Paired t-test: t={t_stat:.2f}, p={p_val:.2e}")
print(f"Cohen's d = {cohen_d:.2f}")

# 绘图
fig, ax = plt.subplots(figsize=(6, 4))
means = [r0_all.mean(), r1_all.mean(), r2_all.mean()]
stds = [r0_all.std(), r1_all.std(), r2_all.std()]
labels = ["Baseline", "GABA_B (2s)", "Post"]

ax.bar(labels, means, yerr=stds, capsize=5, color=['#1f77b4', '#d62728', '#2ca02c'])
ax.set_ylabel("Population rate (Hz)")
ax.set_title("T2.4b: GABA_B network silencing (n=10 seeds)")

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.5, f"{m:.1f}±{s:.1f}", ha='center', fontsize=9)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "f_t24b_stats.png")
plt.savefig(fig_path, dpi=150)
print(f"\nsaved: {fig_path}")