"""T2.1a 等价性冒烟：make_ing(N=100, i_mu=1.0) 必须复现 T2.0 benchmark 的 22.4Hz/46.3Hz。

判据（两条同时过才算移植忠实）：
  rate 偏差 < 5%  且  f0 偏差 < 5%
任何一条 FAIL -> networks.py 移植有量纲/连接错误，停下贴回来。
make_ing 返回三元组 (G, syn_ii, gap)：调用方持有突触引用，防 GC 回收（2026-10-17 教训）。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import start_scope, run, defaultclock, seed, ms, SpikeMonitor
from common.networks import make_ing

N = 100
I_MU = 1.0
SEED = 42
DT_MS = 0.025
T_DISCARD_MS = 500.0
T_MEASURE_MS = 2000.0


def main():
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(SEED)
    G, syn_ii, gap = make_ing(N, i_mu=I_MU, seed=SEED)
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

    ok_rate = abs(rate - 22.4) / 22.4 < 0.05
    ok_f0 = abs(f_peak - 46.3) / 46.3 < 0.05

    print(f"N={N}, i_mu={I_MU}, seed={SEED}")
    print(f"  rate    : {rate:.2f} Hz   (基准 22.4, 偏差 {abs(rate-22.4)/22.4*100:.1f}%)")
    print(f"  f0      : {f_peak:.2f} Hz   (基准 46.3, 偏差 {abs(f_peak-46.3)/46.3*100:.1f}%)")
    print(f"  f0/rate : {f_peak/rate:.2f}")
    print(f"  gamma%  : {gamma_pow:.3f}")
    print(f"  等价性  : rate {'PASS' if ok_rate else 'FAIL'}, f0 {'PASS' if ok_f0 else 'FAIL'}")


if __name__ == "__main__":
    main()