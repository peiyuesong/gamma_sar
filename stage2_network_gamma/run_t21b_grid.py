"""T2.1b 三轴网格：g_Kv3 x I_MU x g_Kdr -> (rate, f0, gamma%, sync_r)。

科学问题（三个，一张网格全回答）：
  Q1 g_Kv3 是不是网络 gamma 的"频率旋钮"？
  Q2 gamma 窗口有多宽？（I_MU 精细档找上下沿）
  Q3 I_MU=2.0 是真同步 gamma 还是异步伪峰？（sync_r 裁决）

新指标 sync_r：1ms 分箱群体 spike 矩阵（N x 2000）成对 Pearson 相关均值。
  sync_r > 0.1 且 f0/rate > 1.5 -> 隔周期 gamma（经典 ING）
  sync_r > 0.1 且 f0/rate ~ 1   -> 每周期同步 gamma（同样是真振荡）
  sync_r < 0.05                 -> 异步放电，f0 是伪峰

细胞电导注入：临时改 PARAMS['cell'] -> make_ing -> finally 恢复
（make_pv_neuron 建组时把电导烘成常数，创建后恢复全局安全；不动 networks.py）。
"""
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import start_scope, run, defaultclock, seed, ms, SpikeMonitor
from common.networks import make_ing
from common.params import PARAMS

N = 100
SEED = 42
DT_MS = 0.025
T_DISCARD_MS = 500.0
T_MEASURE_MS = 2000.0

G_KV3_LIST = [0.0, 10.0, 20.0, 40.0]
I_MU_LIST = [0.6, 0.8, 1.0, 1.2, 1.4, 2.0]
G_KDR_LIST = [10.0, 20.0, 40.0]


def one_run(g_kv3, g_kdr, i_mu):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(SEED)

    cell = PARAMS["cell"]
    saved = (cell.g_Kv3, cell.g_Kdr)
    cell.g_Kv3, cell.g_Kdr = g_kv3, g_kdr
    try:
        G, syn_ii, gap = make_ing(N, i_mu=i_mu, seed=SEED)
    finally:
        cell.g_Kv3, cell.g_Kdr = saved

    sp = SpikeMonitor(G)
    run(T_DISCARD_MS * ms, report=None)
    n_pre = len(sp.t)
    run(T_MEASURE_MS * ms, report=None)

    all_t = np.array(sp.t / ms)
    all_i = np.array(sp.i, dtype=int)
    t_meas = all_t[n_pre:] - T_DISCARD_MS
    i_meas = all_i[n_pre:]
    keep = (t_meas >= 0) & (t_meas <= T_MEASURE_MS)
    t_meas = t_meas[keep]
    i_meas = i_meas[keep]

    rate = len(t_meas) / (N * (T_MEASURE_MS / 1000.0))

    bins = np.arange(0, T_MEASURE_MS + 1, 1.0)
    hist, _ = np.histogram(t_meas, bins=bins)
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, d=1e-3)
    band = (freqs >= 20) & (freqs <= 80)
    f0 = freqs[band][np.argmax(spec[band])] if spec[band].sum() > 0 else 0.0
    gamma_frac = spec[band].sum() / spec.sum() if spec.sum() > 0 else 0.0

    if len(t_meas) > N:
        M, _, _ = np.histogram2d(i_meas, t_meas,
                                 bins=[np.arange(N + 1), bins])
        Mc = M - M.mean(axis=1, keepdims=True)
        valid = Mc.std(axis=1) > 0
        if valid.sum() > 1:
            Mv = Mc[valid]
            Mv = Mv / (Mv.std(axis=1, keepdims=True) + 1e-12)
            C = (Mv @ Mv.T) / Mv.shape[1]
            nv = C.shape[0]
            sync_r = (C.sum() - np.trace(C)) / (nv * (nv - 1))
        else:
            sync_r = 0.0
    else:
        sync_r = 0.0

    return rate, f0, gamma_frac, float(sync_r)


def classify(rate, f0, sync_r):
    if sync_r < 0.05:
        return "async"
    if rate > 0 and abs(f0 / rate - 1.0) < 0.25:
        return "sync-1:1"
    if rate > 0 and f0 / rate > 1.5:
        return "ING-2:1"
    return "sync?"


def main():
    total = len(G_KV3_LIST) * len(I_MU_LIST) * len(G_KDR_LIST)
    print(f"T2.1b grid: g_Kv3 {G_KV3_LIST} x I_MU {I_MU_LIST} x g_Kdr {G_KDR_LIST}")
    print(f"= {total} runs, N={N}, est 10-20 min wall\n")
    rows = []
    for g_kdr in G_KDR_LIST:
        for g_kv3 in G_KV3_LIST:
            for i_mu in I_MU_LIST:
                t0 = time.perf_counter()
                rate, f0, gf, sr = one_run(g_kv3, g_kdr, i_mu)
                wall = time.perf_counter() - t0
                cls = classify(rate, f0, sr)
                print(f"gKv3={g_kv3:>4.0f} gKdr={g_kdr:>4.0f} i={i_mu:>3.1f} | "
                      f"rate={rate:6.1f} f0={f0:5.1f} "
                      f"f0/r={f0/rate if rate>0 else 0:4.2f} "
                      f"g%={gf:5.3f} sync={sr:5.3f} [{cls:7}] ({wall:.0f}s)")
                rows.append((g_kv3, g_kdr, i_mu, rate, f0,
                             f0 / rate if rate > 0 else 0, gf, sr, cls, wall))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "t21b_grid.csv")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write("g_kv3,g_kdr,i_mu,rate_hz,f0_hz,f0_over_rate,gamma_frac,"
                "sync_r,class,wall_s\n")
        for r in rows:
            f.write(f"{r[0]:.0f},{r[1]:.0f},{r[2]:.1f},{r[3]:.2f},{r[4]:.2f},"
                    f"{r[5]:.3f},{r[6]:.4f},{r[7]:.4f},{r[8]},{r[9]:.1f}\n")
    print(f"\nCSV: {out}")

    print("\n=== 三问快答 ===")
    base = sorted([r for r in rows if r[1] == 20.0 and abs(r[2] - 1.0) < 1e-9],
                  key=lambda r: r[0])
    print("Q1 频率旋钮 (gKdr=20, i=1.0): " +
          " | ".join(f"gKv3={r[0]:.0f}: f0={r[4]:.1f} sync={r[7]:.2f}" for r in base))
    g40 = sorted([r for r in rows if r[0] == 40.0 and r[1] == 20.0],
                 key=lambda r: r[2])
    print("Q2 窗口宽度 (gKv3=40, gKdr=20): " +
          " | ".join(f"i={r[2]:.1f}: {r[8]}" for r in g40))
    i2 = sorted([r for r in rows if abs(r[2] - 2.0) < 1e-9 and r[1] == 20.0],
                key=lambda r: r[0])
    print("Q3 i=2.0 裁决 (gKdr=20): " +
          " | ".join(f"gKv3={r[0]:.0f}: sync={r[7]:.2f} [{r[8]}]" for r in i2))


if __name__ == "__main__":
    main()