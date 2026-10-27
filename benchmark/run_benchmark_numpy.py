"""任务 0.4b benchmark：ING 200 PV 神经元，2s 生物时间墙钟测量。

实现说明（重要）：
- 本沙盒无 brian2（environment.yml 环境外），此为 **numpy 参考实现**，
  方程与常数逐行对齐 common/cells.py + common/params.py（单一事实源），
  正式验收前需在 brian2 环境内复测（make benchmark）。
- 门控动力学与 cells.py 一致（无 WB96 原文 φ=5 温度系数——cells.py 未含，
  记为 T1.1 前置检查项，见 report.md）。
- 突触为动力学占位（指数衰减 s，τ=10ms，E=-75mV），T2.1 再换饱和动力学；
  本脚本仅服务 0.4b 墙钟分档决策。
"""
import os
import sys
import time
import numpy as np

# repo root（与调用时的 cwd / 相对路径无关）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import PARAMS  # noqa: E402

# ---- 参数（全部取自 params.py，此处无硬编码参数）----
P = PARAMS["cell"]
N = PARAMS["net"].N_PV
DT = 0.025            # ms（对齐 cells.py 的 dt=0.025ms）
T_DISCARD = 500.0     # ms 预热
T_MEASURE = 2000.0    # ms 计时段（0.4b 口径）
P_II = PARAMS["net"].p_connect_II
P_GAP = 0.1           # networks.py 同款
G_GABA_TOTAL = 0.1    # mS/cm²，网络平均总抑制（T2.1 标定前占位）
G_GAP_TOTAL = 0.2     # mS/cm²（扫描轴 2 的中段）
I_MU = 1.0            # μA/cm²，T2.1 f0 调谐前占位
HETERO = PARAMS["net"].I_hetero
TAU_SYN = PARAMS["net"].tau_syn_GABA_A
E_GABA = -75.0
SEED = 42


def gates(v):
    am = -0.1 * (v + 35.0) / (np.expm1(-(v + 35.0) / 10.0))
    bm = 4.0 * np.exp(-(v + 60.0) / 18.0)
    ah = 0.07 * np.exp(-(v + 58.0) / 20.0)
    bh = 1.0 / (np.exp(-(v + 28.0) / 10.0) + 1.0)
    an = -0.01 * (v + 34.0) / (np.expm1(-(v + 34.0) / 10.0))
    bn = 0.125 * np.exp(-(v + 44.0) / 80.0)
    return am, bm, ah, bh, an, bn


def run_segment(v, h, n, m, p, q, s, I_app, A_ii, A_gap, g_ii_edge, g_gap_edge,
                nsteps, spikes, t0):
    """推进 nsteps，就地更新状态；spikes 收 (t_ms, neuron_idx)。"""
    tau_p = P.tau_p_min + P.tau_p_slow / (1.0 + np.exp((v + 10.0) / 15.0))
    gap_in_sum = (A_gap * g_gap_edge).sum(axis=0)      # Σ_j g_ij（对每个 i）
    for k in range(nsteps):
        am, bm, ah, bh, an, bn = gates(v)
        m += DT * (am * (1 - m) - bm * m)
        h += DT * (ah * (1 - h) - bh * h)
        n += DT * (an * (1 - n) - bn * n)
        p_inf = 1.0 / (1.0 + np.exp(-(v - P.V12_p) / P.k_p))
        p += DT * (p_inf - p) / tau_p
        q_inf = P.q_inf_a + (1 - P.q_inf_a) / (1.0 + np.exp((v + 16.0) / P.q_inf_k))
        q += DT * (q_inf - q) / P.tau_q

        I_leak = P.g_leak * (P.E_leak - v)
        I_Na = P.g_Na * m**3 * h * (P.E_Na - v)
        I_Kdr = P.g_Kdr * n**4 * (P.E_K - v)
        I_Kv3 = P.g_Kv3 * p**2 * q * (P.E_K - v)
        # 内向为正约定（与 cells.py 一致）：I = g * (E − v)
        I_gaba = (A_ii * (g_ii_edge * s)[None, :]).sum(axis=1) * (E_GABA - v)
        # gap 连续扩散耦合：I_i = Σ_j g_ij (v_j − v_i) = g(A@v)_i − g·deg_i·v_i
        I_gap = g_gap_edge * (A_gap @ v) - gap_in_sum * v

        v_old = v
        v = v + DT / P.C * (I_leak + I_Na + I_Kdr + I_Kv3 + I_gaba + I_gap + I_app)

        s = s + DT * (-s / TAU_SYN)
        fired = (v_old < 0.0) & (v >= 0.0)
        if fired.any():
            idx = np.nonzero(fired)[0]
            # 饱和动力学（对齐 cells.py 两态 r 的饱和性质）：s ∈ [0,1]
            s = np.minimum(s + A_ii[:, idx].sum(axis=1), 1.0)
            for i in idx:
                spikes.append((t0 + k * DT, int(i)))
    return v, h, n, m, p, q, s, t0 + nsteps * DT


def main():
    rng = np.random.default_rng(SEED)
    A_ii = (rng.random((N, N)) < P_II).astype(float)
    np.fill_diagonal(A_ii, 0.0)
    A_gap = (rng.random((N, N)) < P_GAP).astype(float)
    np.fill_diagonal(A_gap, 0.0)
    g_ii_edge = G_GABA_TOTAL * N / max(A_ii.sum(), 1.0)   # 每边电导，使每细胞平均总量≈G_GABA_TOTAL
    g_gap_edge = G_GAP_TOTAL * N / max(A_gap.sum(), 1.0)
    I_app = I_MU * (1.0 + HETERO * rng.standard_normal(N))
    I_app = np.clip(I_app, 0.0, None)

    v = np.full(N, -65.0); h = np.full(N, 0.6); n = np.full(N, 0.3)
    m = np.full(N, 0.05); p = np.zeros(N); q = np.zeros(N); s = np.zeros(N)

    spikes = []
    v, h, n, m, p, q, s, t0 = run_segment(
        v, h, n, m, p, q, s, I_app, A_ii, A_gap, g_ii_edge, g_gap_edge,
        int(T_DISCARD / DT), spikes, 0.0)
    n_pre = len(spikes)

    t_start = time.perf_counter()
    v, h, n, m, p, q, s, t_end = run_segment(
        v, h, n, m, p, q, s, I_app, A_ii, A_gap, g_ii_edge, g_gap_edge,
        int(T_MEASURE / DT), spikes, t0)
    wall = time.perf_counter() - t_start
    spikes = np.array(spikes[n_pre:], dtype=float)

    rate = len(spikes) / (N * (T_MEASURE / 1000.0))
    bins = np.arange(0, T_MEASURE + 1, 1.0)
    rate_hist, _ = np.histogram(spikes[:, 0] - t0, bins=bins)
    r = rate_hist - rate_hist.mean()
    nfft = 8192
    spec = np.abs(np.fft.rfft(r, nfft))**2
    freqs = np.fft.rfftfreq(nfft, d=1e-3)
    band = (freqs >= 20) & (freqs <= 80)
    f_peak = freqs[band][np.argmax(spec[band])]
    gamma_pow = spec[band].sum() / spec.sum()

    wall_2h_s = wall * (7200.0 / (T_MEASURE / 1000.0))
    print("=" * 60)
    print(f"单次 {T_MEASURE/1000:.1f}s 生物时间墙钟 : {wall:8.2f} s")
    print(f"外推单次 2h 生物时间墙钟 : {wall_2h_s:8.1f} s = {wall_2h_s/3600:.3f} h")
    print(f"平均放电率 : {rate:8.1f} Hz")
    print(f"f0 峰频(PSD, 20-80Hz): {f_peak:8.1f} Hz")
    print(f"γ带功率占比 : {gamma_pow:8.3f}")
    print(f"spikes 计数 : {len(spikes)}")


if __name__ == "__main__":
    main()
