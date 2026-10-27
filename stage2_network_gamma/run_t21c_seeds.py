"""T2.1c：g_Kv3 延伸轴(至80) + 模式边界 seed 稳定性。

判决三假说：
  频率旋钮 -> f0 随 g_Kv3 过 40 后继续单调升
  阈值开关 -> f0 钉死 46 或全崩 1:1
  模式锁   -> 2:1 占比升，f0 本身不动

网格（g_kdr=20 为主）：
  A. i=1.0:  gKv3 in {0,10,20,30,40,60,80}   <- 判决轴（你的建议）
  B. i=0.8/1.2: gKv3 in {0,40,80}            <- 高位 Kv3 能否跨驱动保/救 2:1
  C. gKv3=40,i=1.0: gKdr in {10,15,20,25,40} <- 细化非单调(发现B)
  D. gKv3=40,gKdr=20: i in {0.65,0.70,0.75}  <- 下沿定位
20 行 x 5 seeds = 100 runs, ~12 min。
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
SEEDS = [1, 7, 13, 27, 42]

ROWS = (
    [(g, 20.0, 1.0) for g in [0, 10, 20, 30, 40, 60, 80]] +
    [(g, 20.0, 0.8) for g in [0, 40, 80]] +
    [(g, 20.0, 1.2) for g in [0, 40, 80]] +
    [(40.0, k, 1.0) for k in [10, 15, 20, 25, 40]] +
    [(40.0, 20.0, i) for i in [0.65, 0.70, 0.75]]
)


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
    gf = spec[band].sum() / spec.sum() if spec.sum() > 0 else 0.0

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
    return rate, f0, gf, float(sr), cls


def main():
    print(f"T2.1c: {len(ROWS)} rows x {len(SEEDS)} seeds = {len(ROWS)*len(SEEDS)} runs\n")
    per_row = defaultdict(list)
    out_rows = []
    for (g, k, i) in ROWS:
        for sd in SEEDS:
            t0 = time.perf_counter()
            rate, f0, gf, sr, cls = one_run(g, k, i, sd)
            wall = time.perf_counter() - t0
            per_row[(g, k, i)].append((rate, f0, sr, cls))
            out_rows.append((g, k, i, sd, rate, f0, gf, sr, cls, wall))
        rs = per_row[(g, k, i)]
        f0s = [r[1] for r in rs]
        p21 = sum(1 for r in rs if r[3] == "ING-2:1") / len(rs)
        p11 = sum(1 for r in rs if r[3] == "sync-1:1") / len(rs)
        pa = sum(1 for r in rs if r[3] == "async") / len(rs)
        print(f"gKv3={g:>4.0f} gKdr={k:>4.0f} i={i:>4.2f} | "
              f"f0={np.mean(f0s):5.1f}+-{np.std(f0s):4.1f} "
              f"2:1={p21:.0%} 1:1={p11:.0%} async={pa:.0%}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "t21c_seeds.csv")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write("g_kv3,g_kdr,i_mu,seed,rate_hz,f0_hz,gamma_frac,sync_r,class,wall_s\n")
        for r in out_rows:
            f.write(f"{r[0]:.0f},{r[1]:.0f},{r[2]:.2f},{r[3]},{r[4]:.2f},{r[5]:.2f},"
                    f"{r[6]:.4f},{r[7]:.4f},{r[8]},{r[9]:.1f}\n")
    print(f"\nCSV: {out}")

    print("\n=== 判决读数 ===")
    a = per_row
    print("A 轴 (i=1.0, gKdr=20):")
    for g in [0, 10, 20, 30, 40, 60, 80]:
        rs = a[(g, 20.0, 1.0)]
        f0m = np.mean([r[1] for r in rs])
        p21 = sum(1 for r in rs if r[3] == "ING-2:1") / len(rs)
        print(f"  gKv3={g:>2.0f}: f0={f0m:5.1f} Hz  2:1占比={p21:.0%}")


if __name__ == "__main__":
    main()