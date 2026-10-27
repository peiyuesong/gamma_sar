"""双稳真假验证 v2：固定网络（seed=42），只变初始条件；逐 run 容错。

v2 改动：删 v0=-45（极端初值触发 TM 门控可去奇点，非 basin 探针）；
        逐 run try/except（T2.5+ 长跑工程标准）。
判据：同网络 + 不同初值 -> 落不同盆地 = 真双稳。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import start_scope, run, defaultclock, seed, ms, mV, SpikeMonitor
from common.networks import make_ing
from common.params import PARAMS

N = 100
SEED = 42
DT_MS = 0.025
T_DISCARD_MS = 500.0
T_MEASURE_MS = 2000.0

V0_LIST = [-65.0, -60.0, -55.0, -50.0]
S0_LIST = [0.5, 0.8, 1.0]


def one_run(v0, s0):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(SEED)
    cell = PARAMS["cell"]
    saved = (cell.g_Kv3, cell.g_Kdr)
    cell.g_Kv3, cell.g_Kdr = 40.0, 20.0
    try:
        G, syn_ii, gap = make_ing(N, i_mu=1.0, seed=SEED)
    finally:
        cell.g_Kv3, cell.g_Kdr = saved

    G.v = v0 * mV
    G.s = s0

    sp = SpikeMonitor(G)
    run(T_DISCARD_MS * ms, report=None)
    n_pre = len(sp.t)
    run(T_MEASURE_MS * ms, report=None)

    all_t = np.array(sp.t / ms)
    t = all_t[n_pre:] - T_DISCARD_MS
    t = t[(t >= 0) & (t <= T_MEASURE_MS)]
    rate = len(t) / (N * (T_MEASURE_MS / 1000.0))

    bins = np.arange(0, T_MEASURE_MS + 1, 1.0)
    hist, _ = np.histogram(t, bins=bins)
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, d=1e-3)
    band = (freqs >= 20) & (freqs <= 80)
    f0 = freqs[band][np.argmax(spec[band])] if spec[band].sum() > 0 else 0.0

    if rate <= 0:
        return rate, f0, "async"
    if abs(f0 / rate - 1.0) < 0.25:
        return rate, f0, "sync-1:1"
    if f0 / rate > 1.5:
        return rate, f0, "ING-2:1"
    return rate, f0, "sync?"


def main():
    print("双稳验证 v2: (gKv3=40, gKdr=20, i=1.0), seed=42 固定, "
          "v0 x s0 -> 12 runs, per-run fault tolerant\n")
    results = {}
    for v0 in V0_LIST:
        for s0 in S0_LIST:
            try:
                t0 = time.perf_counter()
                rate, f0, cls = one_run(v0, s0)
                wall = time.perf_counter() - t0
                results[(v0, s0)] = cls
                print(f"v0={v0:>5.0f} s0={s0:.1f} | rate={rate:6.1f} "
                      f"f0={f0:5.1f} [{cls:7}] ({wall:.0f}s)")
            except Exception as e:
                results[(v0, s0)] = "CRASH"
                print(f"v0={v0:>5.0f} s0={s0:.1f} | CRASH ({type(e).__name__})")

    modes = sorted({c for c in results.values() if c != "CRASH"})
    print("\n=== 判决 ===")
    print(f"出现的模式集合: {modes} (CRASH {sum(1 for c in results.values() if c == 'CRASH')} 个)")
    if len(modes) >= 2:
        print("-> 真双稳：初始条件决定盆地，'刺激可翻转'叙事成立")
    else:
        print("-> 网络实现异质性：叙事改为'个体响应概率分布'")


if __name__ == "__main__":
    main()