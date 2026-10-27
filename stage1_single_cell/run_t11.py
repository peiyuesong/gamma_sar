"""T1.1：单神经元三联图复跑 + WB96 φ=5 温度系数 A/B。

验收线（路线图 3.3）：含 Kv3 强驱动放电 >200Hz、AP 半宽 ≤0.6ms、2s 末段 V ≥ −55mV。
对照雏形值：308Hz、半宽 0.46ms、末段 −49mV（原型口径；半宽/末段测量口径见
params_v1.4.md 修订沿革，数值不作逐位追平）。

用法（mleDSar 环境）：
    python stage1_single_cell/run_t11.py
输出：φ × I 网格的 rate / 半宽 / 末段电位表。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (  # noqa: E402
    start_scope, run, defaultclock, seed, ms, mV, uA, cm2,
    SpikeMonitor, StateMonitor,
)
from common.cells import make_pv_neuron  # noqa: E402

I_LIST = [15.0, 30.0, 45.0, 60.0]  # μA/cm²（原型 1.5–6nA @10000μm²；1nA ≡ 10μA/cm²）
PHI_LIST = [1.0]                   # 原型无温度系数（2026-10-10 裁定）；φ 降为 T1.4 敏感性项
DUR_MS = 2000.0
DT_MS = 0.025
V_END_WINDOW_MS = 100.0           # 末段电位 = 最后 100ms 均值（相位平均，非单点）


def measure(v_arr, dt_ms, rate_hz):
    """稳健版：局部最大值 + 局部基线半幅 + 峰幅过滤。

    半宽口径：局部最小（前 3ms 内）到峰值的中点电平测宽，取所有稳态尖峰中位数。
    末段电位：最后 100ms 均值（单点采样在 >200Hz 下是相位抽签，不作数）。
    """
    n_end = int(V_END_WINDOW_MS / dt_ms)
    v_end = float(v_arr[-n_end:].mean())
    half_idx = len(v_arr) // 2
    seg = v_arr[half_idx:]
    if len(seg) < 10:
        return np.nan, v_end
    is_peak = np.zeros(len(seg), dtype=bool)
    is_peak[1:-1] = (seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])
    peaks = np.nonzero(is_peak & (seg > -30.0))[0]
    if len(peaks) == 0:
        return np.nan, v_end
    widths = []
    for pi in peaks:
        peak = seg[pi]
        j0 = max(0, pi - int(3.0 / dt_ms))
        local_min = seg[j0:pi + 1].min()
        amp = peak - local_min
        if amp < 20.0:
            continue
        half_level = local_min + 0.5 * amp
        j = pi
        while j > 0 and seg[j] >= half_level:
            j -= 1
        if j == 0:
            continue
        k = pi
        while k < len(seg) - 1 and seg[k] >= half_level:
            k += 1
        if k == len(seg) - 1:
            continue
        widths.append((k - j) * dt_ms)
    if not widths:
        return np.nan, v_end
    return float(np.median(widths)), v_end


def one_run(phi, i_uA_cm2):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(0)
    # 原型 spike 语义：reset=None 自然复极 + 条件不应期（v 回 −20 以下才再触发，防双计数）
    G = make_pv_neuron(1, reset=None, refractory="v > -20.0 * mV")
    G.phi = phi
    G.I_inj = i_uA_cm2 * (uA / cm2)
    mon = StateMonitor(G, "v", record=True)
    sp = SpikeMonitor(G)
    run(DUR_MS * ms, report=None)
    rate = len(sp.t) / (DUR_MS / 1000.0)
    width_ms, v_end = measure(np.array(mon.v[0] / mV), DT_MS, rate)
    return rate, width_ms, v_end


def main():
    print(f"{'phi':>4} {'I(uA/cm2)':>10} {'rate(Hz)':>9} {'half(ms)':>9} {'v_end(mV)':>10}")
    print("-" * 48)
    for phi in PHI_LIST:
        for i_inj in I_LIST:
            rate, w, vend = one_run(phi, i_inj)
            print(f"{phi:>4.1f} {i_inj:>10.1f} {rate:>9.1f} {w:>9.3f} {vend:>10.1f}")
        print("-" * 48)
    print()
    print("验收线（原型血统，2026-10-10 裁定）: f-I >200Hz@强驱动 | 有无 Kv3 半宽对比 ≥2×（Erisir 锚）")
    print("原型参考: 6nA≡60μA/cm²；触发平均半宽 1.7 vs 4.0ms；末段 −49 vs −34mV")


if __name__ == "__main__":
    main()