"""
【已弃用 2026-10-10】旧架构遗物：主 I_Na 不乘 s 时代的 g_Na_slow 扫描。
现架构主 I_Na 已乘 s，跑本会双倍失活。T1.2 验收以 run_t12b / run_t12c 为准。

T1.2：Na 慢失活——无 Kv3 组强注入下阻滞/崩溃、含 Kv3 组存活。

验收（路线图 3.2 / 3.3）：g_Kv3=0 组强驱动 → 放电停止或进入去极化阻滞
（rate 骤降 / v_end 持续去极化）；g_Kv3=20 组同驱动 → 维持放电。
φ 用 params 默认值（5.0）；g_Na_slow × I 为扫描轴，τ_s 用基准 tau_s_na。

用法（mleDSar 环境）：
    python stage1_single_cell/run_t12.py
输出：g_Kv3 × g_Na_slow × I 网格的 rate / 半宽 / 末段电位表。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (  # noqa: E402
    start_scope, run, defaultclock, seed, ms, mV, mS, cm2, uA,
    SpikeMonitor, StateMonitor,
)
from common.cells import make_pv_neuron  # noqa: E402
from common.params import PARAMS  # noqa: E402
from stage1_single_cell.run_t11 import measure  # noqa: E402

G_KV3_LIST = [0.0, 40.0]          # mS/cm²（0 = 无 Kv3 组；40 = 原型基准）
G_NASLOW_LIST = [0.0, 0.5, 1.0]   # mS/cm²（0 = 基线对照）
I_LIST = [30.0, 45.0, 60.0]       # μA/cm²（原型 3–6nA 档）
DUR_MS = 2000.0
DT_MS = 0.025


def one_run(g_kv3, g_na_slow, i_uA_cm2):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(0)
    # reset=None + 条件不应期（原型 spike 语义）
    G = make_pv_neuron(1, reset=None, refractory="v > -20.0 * mV")
    G.g_Kv3 = g_kv3 * (mS / cm2)
    G.g_Na_slow = g_na_slow * (mS / cm2)
    G.I_inj = i_uA_cm2 * (uA / cm2)
    mon = StateMonitor(G, "v", record=True)
    sp = SpikeMonitor(G)
    run(DUR_MS * ms, report=None)
    rate = len(sp.t) / (DUR_MS / 1000.0)
    width_ms, v_end = measure(np.array(mon.v[0] / mV), DT_MS, rate)
    return rate, width_ms, v_end


def blocked(rate, v_end):
    """阻滞判据：放电近乎停止，或末段持续去极化（去极化阻滞）。"""
    return rate < 5.0 or v_end > -35.0


def main():
    print(f"{'gKv3':>5} {'gNasl':>6} {'I':>4} {'rate(Hz)':>9} {'half(ms)':>9} {'v_end(mV)':>10} {'判定':>6}")
    print("-" * 56)
    for g_kv3 in G_KV3_LIST:
        for g_ns in G_NASLOW_LIST:
            for i_inj in I_LIST:
                rate, w, vend = one_run(g_kv3, g_ns, i_inj)
                verdict = "BLOCK" if blocked(rate, vend) else "OK"
                print(f"{g_kv3:>5.1f} {g_ns:>6.2f} {i_inj:>4.1f} {rate:>9.1f} "
                      f"{w:>9.3f} {vend:>10.1f} {verdict:>6}")
            print("-" * 56)
    print()
    print("期望：gKv3=0 组在 gNasl>0 + 强 I 下出现 BLOCK；gKv3=20 组全 OK")
    print("若 gKv3=0 组不崩 → 增大 gNasl 或延长 DUR；若 gKv3=20 组也崩 → 参数回查")


if __name__ == "__main__":
    main()