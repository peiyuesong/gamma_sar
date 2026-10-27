"""任务 0.4b benchmark（brian2 正式版）：ING 200 PV，2s 生物时间墙钟。

与 run_benchmark_numpy.py 逐项对齐（同一模型，两种实现）：
  - GABA_A 电导型饱和突触（τ=10ms，E=-75mV，s∈[0,1]逐边归一化）——非电压踢
  - gap 逐边电导归一化（每细胞期望总量 ≈ g_gap_default）
  - 不放电重置（threshold 仅记数 + refractory 0.7ms 防重复计数，对齐 numpy 的越零检测）
本脚本是分档决策的正式输入。降维：$env:ING_N = "100"
"""
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
N = int(os.environ.get("ING_N", str(NP.N_PV)))
DT_MS = 0.025
T_DISCARD_MS = 500.0
T_MEASURE_MS = 2000.0
P_II = NP.p_connect_II
P_GAP = 0.1                       # 与 numpy 版同款占位（T2.1 标定）
G_GABA_TOTAL = 0.1                # 占位：ING 平均总抑制（Wang-Buzsáki 1996 量级）
G_GAP_TOTAL = NP.g_gap_default    # 0.2 mS/cm²
E_GABA = -75.0                    # 占位：Wang-Buzsáki 1996 反转电位
TAU_GABA = NP.tau_syn_GABA_A      # 10 ms
I_MU_uA_cm2 = 1.0                 # 占位驱动（T2.1 f0 调谐）
HETERO = NP.I_hetero
SEED = 42


def main():
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(SEED)

    rng = np.random.default_rng(SEED)

    G = make_pv_neuron(N, reset=None, refractory="v > -20.0 * mV")
    I_app = I_MU_uA_cm2 * (1.0 + HETERO * rng.standard_normal(N))
    I_app = np.clip(I_app, 0.0, None)
    G.I_inj = I_app * (uA / cm2)

    # GABA_A：电导型 + 饱和（s 钳制 [0,1]，对齐 numpy 参考实现）
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

    # gap：扩散耦合，逐边归一化（每细胞期望总量 ≈ G_GAP_TOTAL）
    g_gap_each = G_GAP_TOTAL / (N * P_GAP)
    gap = Synapses(G, G,
                   model="""
        I_gap_post = g_gap * (v_pre - v_post) : amp/meter**2 (summed)
        g_gap : siemens/meter**2 (constant)
        """)
    gap.connect(p=P_GAP)
    gap.g_gap = g_gap_each * mS / cm2

    spike_mon = SpikeMonitor(G)

    print(f"N = {N}, dt = {DT_MS} ms, T_discard = {T_DISCARD_MS} ms, T_measure = {T_MEASURE_MS} ms")
    run(T_DISCARD_MS * ms, report=None)
    n_pre = len(spike_mon.t)

    t0 = time.perf_counter()
    run(T_MEASURE_MS * ms, report=None)
    wall = time.perf_counter() - t0

    all_t = np.array(spike_mon.t / ms)
    spike_t = all_t[n_pre:] - T_DISCARD_MS
    spike_t = spike_t[(spike_t >= 0) & (spike_t <= T_MEASURE_MS)]

    rate = len(spike_t) / (N * (T_MEASURE_MS / 1000.0))

    bins = np.arange(0, T_MEASURE_MS + 1, 1.0)
    rate_hist, _ = np.histogram(spike_t, bins=bins)
    r = rate_hist - rate_hist.mean()
    nfft = 8192
    spec = np.abs(np.fft.rfft(r, nfft))**2
    freqs = np.fft.rfftfreq(nfft, d=1e-3)
    band = (freqs >= 20) & (freqs <= 80)
    f_peak = freqs[band][np.argmax(spec[band])]
    gamma_pow = spec[band].sum() / spec.sum()

    wall_2h_s = wall * (7200.0 / (T_MEASURE_MS / 1000.0))
    print("=" * 60)
    print(f"单次 {T_MEASURE_MS/1000:.1f}s 生物时间墙钟 : {wall:8.2f} s")
    print(f"外推单次 2h 生物时间墙钟 : {wall_2h_s:8.1f} s = {wall_2h_s/3600:.3f} h")
    print(f"平均放电率 : {rate:8.1f} Hz")
    print(f"f0 峰频(PSD, 20-80Hz): {f_peak:8.1f} Hz")
    print(f"gamma 带功率占比 : {gamma_pow:8.3f}")
    print(f"spikes 计数 : {len(spike_t)}")

    # ---- 复现图（README/MS0 用）：raster + PSD ----
    spike_i = np.array(spike_mon.i[n_pre:], dtype=int)
    keep = (spike_t >= 0) & (spike_t <= T_MEASURE_MS)
    spike_t, spike_i = spike_t[keep], spike_i[keep]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    sub = spike_i < 50
    axes[0].scatter(spike_t[sub], spike_i[sub], s=2, c="k")
    axes[0].set(xlabel="t (ms)", ylabel="neuron", title=f"ING raster ({min(50, N)}/{N} PV)")
    band_plot = (freqs >= 0) & (freqs <= 100)
    axes[1].plot(freqs[band_plot], spec[band_plot] / spec.max(), lw=1)
    axes[1].axvspan(20, 80, color="orange", alpha=0.15, label="γ (20–80 Hz)")
    axes[1].axvline(f_peak, color="r", ls="--", label=f"f0 = {f_peak:.1f} Hz")
    axes[1].set(xlabel="frequency (Hz)", ylabel="normalized PSD", title="Population rate PSD")
    axes[1].legend()
    out = os.path.join(fig_dir, "benchmark_psd.png")
    fig.savefig(out, dpi=150)
    print(f"复现图已保存: {out}")


if __name__ == "__main__":
    main()