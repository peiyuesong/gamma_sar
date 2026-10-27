"""T2.1 前置诊断：rate 随驱动 I_MU 的变化。

背景：T2.0 显示新架构下 rate 掉到 22.6 Hz（旧 64 Hz），
怀疑主 Na×s 使单细胞自激能力下降，需要更高驱动。
本脚本扫 I_MU ∈ {1, 2, 4, 8} × N ∈ {100}，看 rate/f0 恢复情况。

输出：终端表 + CSV benchmark/diag_rate_vs_drive.csv
"""
import io
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (
    Synapses, SpikeMonitor, run, start_scope, defaultclock, seed,
    ms, mV, mS, cm2, uA, clip,
)
from common.cells import make_pv_neuron
from common.params import PARAMS

NP = PARAMS["net"]
N = 100
DT_MS = 0.025
T_DISCARD_MS = 500.0
T_MEASURE_MS = 2000.0
P_II = NP.p_connect_II
P_GAP = 0.1
G_GABA_TOTAL = 0.1
G_GAP_TOTAL = NP.g_gap_default
E_GABA = -75.0
TAU_GABA = NP.tau_syn_GABA_A
I_MU_LIST = [1.0, 2.0, 4.0, 8.0]
HETERO = NP.I_hetero
SEED = 42


def one_run(i_mu):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(SEED)
    rng = np.random.default_rng(SEED)
    G = make_pv_neuron(N, reset=None, refractory="v > -20.0 * mV")
    I_app = i_mu * (1.0 + HETERO * rng.standard_normal(N))
    I_app = np.clip(I_app, 0.0, None)
    G.I_inj = I_app * (uA / cm2)

    g_syn_each = G_GABA_TOTAL / (N * P_II)
    syn = Synapses(G, G,
                   model=f"""
        dg/dt = -g / ({TAU_GABA} * ms) : 1
        I_syn_post = g_syn_each * g * ({E_GABA} * mV - v_post) : amp/meter**2 (summed)
        g_syn_each : siemens/meter**2 (constant)
        """,
                   on_pre="""g += 1
        g = clip(g, 0, 1)""")
    syn.connect(p=P_II)
    syn.g_syn_each = g_syn_each * mS / cm2

    g_gap_each = G_GAP_TOTAL / (N * P_GAP)
    gap = Synapses(G, G,
                   model="""
        I_gap_post = g_gap * (v_pre - v_post) : amp/meter**2 (summed)
        g_gap : siemens/meter**2 (constant)
        """)
    gap.connect(p=P_GAP)
    gap.g_gap = g_gap_each * mS / cm2

    sp = SpikeMonitor(G)
    run(T_DISCARD_MS * ms, report=None)
    n_pre = len(sp.t)
    run(T_MEASURE_MS * ms, report=None)

    all_t = np.array(sp.t / ms)
    spike_t = all_t[n_pre:] - T_DISCARD_MS
    spike_t = spike_t[(spike_t >= 0) & (spike_t <= T_MEASURE_MS)]
    rate = len(spike_t) / (N * (T_MEASURE_MS / 1000.0))

    bins = np.arange(0, T_MEASURE_MS + 1, 1.0)
    hist, _ = np.histogram(spike_t, bins=bins)
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, d=1e-3)
    band = (freqs >= 20) & (freqs <= 80)
    f_peak = freqs[band][np.argmax(spec[band])]
    gamma_pow = spec[band].sum() / spec.sum()
    return rate, f_peak, gamma_pow


def main():
    print(f"N={N}, I_MU 扫描 {I_MU_LIST}，各 2s 生物时间")
    print(f"{'I_MU':>6} {'rate(Hz)':>10} {'f0(Hz)':>9} {'f0/rate':>9} {'gamma%':>8}")
    print("-" * 48)
    rows = []
    for i_mu in I_MU_LIST:
        t0 = time.perf_counter()
        rate, f0, gp = one_run(i_mu)
        wall = time.perf_counter() - t0
        ratio = f0 / rate if rate > 0 else float("nan")
        print(f"{i_mu:>6.1f} {rate:>10.1f} {f0:>9.1f} {ratio:>9.2f} {gp:>8.3f}  ({wall:.1f}s)")
        rows.append((i_mu, rate, f0, ratio, gp))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "diag_rate_vs_drive.csv")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write("i_mu,rate_hz,f0_hz,f0_over_rate,gamma_frac\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.4f},{r[4]:.4f}\n")
    print(f"\nCSV: {out}")
    print("\n判读：")
    print("· f0/rate ≈ 1 → f0 是放电率伪峰，非振荡")
    print("· f0/rate ≈ 2-3 → 可能是真 γ 振荡")
    print("· rate 回到 40-80Hz 且 f0/rate >1.5 → 驱动档可进 T2.1")


if __name__ == "__main__":
    main()