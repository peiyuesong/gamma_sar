"""T2.1d 操作图精扫：i x g_Kv3 (x Kdr) -> 2:1 盆地概率 + Wilson CI + 分模式 f0。

升级点（相对 T2.1c）：
  - 20 seeds/行（Wilson 95% CI，比例分辨率 ~18%）
  - f0 报分模式中位数（2:1 与 1:1 分开），消灭双峰混合均值
  - i 步长 0.05，定位窗口边界与 40Hz 运行点
行 = i{0.70..1.20} x gKv3{0,40} (Kdr=20) + Kdr 细化 3 行 = 23 行 x 20 seeds = 460 runs。
"""
import io
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import start_scope, run, defaultclock, seed, ms, SpikeMonitor
from common.networks import make_ing
from common.params import PARAMS

N = 100
DT_MS = 0.025
T_DISCARD_MS = 500.0
T_MEASURE_MS = 2000.0
SEEDS = list(range(20))          # 0..19

I_LIST = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
ROWS = ([(g, 20.0, i) for i in I_LIST for g in (0.0, 40.0)] +
        [(40.0, k, 1.00) for k in (10.0, 15.0, 25.0)])


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def one_run(g_kv3, g_kdr, i_mu, seed_id):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(seed_id)
    cell = PARAMS["cell"]
    saved = (cell.g_Kv3, cell.g_Kdr)
    cell.g_Kv3, cell.g_Kdr = g_kv3, g_kdr
    try:
        G, syn_ii, gap = make_ing(N, i_mu=i_mu, seed=seed_id)
    finally:
        cell.g_Kv3, cell.g_Kdr = saved

    sp = SpikeMonitor(G)
    run(T_DISCARD_MS * ms, report=None)
    n_pre = len(sp.t)
    run(T_MEASURE_MS * ms, report=None)

    all_t = np.array(sp.t / ms)
    all_i = np.array(sp.i, dtype=int)
    t = all_t[n_pre:] - T_DISCARD_MS
    idx = all_i[n_pre:]
    keep = (t >= 0) & (t <= T_MEASURE_MS)
    t, idx = t[keep], idx[keep]

    rate = len(t) / (N * (T_MEASURE_MS / 1000.0))
    bins = np.arange(0, T_MEASURE_MS + 1, 1.0)
    hist, _ = np.histogram(t, bins=bins)
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, d=1e-3)
    band = (freqs >= 20) & (freqs <= 80)
    f0 = freqs[band][np.argmax(spec[band])] if spec[band].sum() > 0 else 0.0

    if len(t) > N:
        M, _, _ = np.histogram2d(idx, t, bins=[np.arange(N + 1), bins])
        Mc = M - M.mean(axis=1, keepdims=True)
        valid = Mc.std(axis=1) > 0
        if valid.sum() > 1:
            Mv = Mc[valid]
            Mv = Mv / (Mv.std(axis=1, keepdims=True) + 1e-12)
            C = (Mv @ Mv.T) / Mv.shape[1]
            nv = C.shape[0]
            sr = (C.sum() - np.trace(C)) / (nv * (nv - 1))
        else:
            sr = 0.0
    else:
        sr = 0.0

    if sr < 0.05:
        cls = "async"
    elif rate > 0 and abs(f0 / rate - 1.0) < 0.25:
        cls = "sync-1:1"
    elif rate > 0 and f0 / rate > 1.5:
        cls = "ING-2:1"
    else:
        cls = "sync?"
    return rate, f0, float(sr), cls


def main():
    n_total = len(ROWS) * len(SEEDS)
    print(f"T2.1d: {len(ROWS)} rows x {len(SEEDS)} seeds = {n_total} runs, "
          f"est {n_total*7//60} min\n")
    long_rows = []
    for (g, k, i) in ROWS:
        t0 = time.perf_counter()
        clss, f0s = [], defaultdict(list)
        for sd in SEEDS:
            rate, f0, sr, cls = one_run(g, k, i, sd)
            clss.append(cls)
            f0s[cls].append(f0)
            long_rows.append((g, k, i, sd, rate, f0, sr, cls))
        n = len(clss)
        k21 = sum(1 for c in clss if c == "ING-2:1")
        k11 = sum(1 for c in clss if c == "sync-1:1")
        ka = sum(1 for c in clss if c == "async")
        p21, lo21, hi21 = wilson(k21, n)
        m21 = np.median(f0s["ING-2:1"]) if f0s["ING-2:1"] else 0.0
        m11 = np.median(f0s["sync-1:1"]) if f0s["sync-1:1"] else 0.0
        wall = time.perf_counter() - t0
        print(f"gKv3={g:>4.0f} gKdr={k:>4.0f} i={i:>4.2f} | "
              f"2:1={p21:.2f}[{lo21:.2f},{hi21:.2f}] 1:1={k11/n:.2f} "
              f"asy={ka/n:.2f} | f0_21={m21:5.1f} f0_11={m11:5.1f} ({wall:.0f}s)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "t21d_fine.csv")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write("g_kv3,g_kdr,i_mu,seed,rate_hz,f0_hz,sync_r,class\n")
        for r in long_rows:
            f.write(f"{r[0]:.0f},{r[1]:.0f},{r[2]:.2f},{r[3]},{r[4]:.2f},"
                    f"{r[5]:.2f},{r[6]:.4f},{r[7]}\n")
    print(f"\nCSV: {out}")
    print("\n=== 读数指南 ===")
    print("· 找 40Hz 运行点：f0_21 中位数接近 40 且 2:1 CI 下沿 >0.5 的行")
    print("· EO/EC 操作点：gKv3=40 列与 gKv3=0 列对比即 T2.6 两档")


if __name__ == "__main__":
    main()