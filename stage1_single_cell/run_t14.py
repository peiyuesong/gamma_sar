"""T1.4 + T1.5：Kv3 门控敏感性 + 非单调性检查。

T1.4（路线图 3.4）：V12_p ±10mV × τ_p 倍乘 {0.25,0.5,1,2,4}（结论方向须不变）
T1.5（路线图 3.5）：τ_p 扫描中若 rate 对去激活速度呈"中间最优"（f_β 类比，
Lien & Jonas 2003），如实记录最优点。
固定 I=45 μA/cm²、g_Kv3=40；附 g_Kv3=0 对照臂（无 Kv3 时门控扫描应为平线）。

用法（mleDSar 环境）：
    python stage1_single_cell/run_t14.py
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
from stage1_single_cell.run_t11 import measure  # noqa: E402

DUR_MS = 2000.0
DT_MS = 0.025
I_FIX = 45.0
V12_LIST = [3.0, 13.0, 23.0]        # ±10mV
TAU_FACTOR_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]


def one_run(g_kv3, v12_p, tau_factor):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(0)
    G = make_pv_neuron(1, reset=None, refractory="v > -20.0 * mV")
    G.g_Kv3 = g_kv3 * (mS / cm2)
    G.V12_p = v12_p * mV
    # τ_p 整体倍乘（min 与 slow 分量同比）
    from common.params import PARAMS
    G.tau_p_min = PARAMS["cell"].tau_p_min * tau_factor * ms
    G.tau_p_slow = PARAMS["cell"].tau_p_slow * tau_factor * ms
    G.I_inj = I_FIX * (uA / cm2)
    mon = StateMonitor(G, "v", record=True)
    sp = SpikeMonitor(G)
    run(DUR_MS * ms, report=None)
    rate = len(sp.t) / (DUR_MS / 1000.0)
    w, vend = measure(np.array(mon.v[0] / mV), DT_MS, rate)
    return rate, w, vend


def main():
    print(f"T1.4/T1.5 @ I={I_FIX:g} μA/cm²，g_Kv3=40（附 0 对照）")
    for g_kv3 in [40.0, 0.0]:
        print(f"\n=== g_Kv3 = {g_kv3:g} ===")
        print(f"{'V12_p':>6} {'τ倍乘':>6} {'rate(Hz)':>9} {'half(ms)':>9} {'v_end(mV)':>10}")
        print("-" * 46)
        for v12 in V12_LIST:
            for tf in TAU_FACTOR_LIST:
                rate, w, vend = one_run(g_kv3, v12, tf)
                print(f"{v12:>6.1f} {tf:>6.2f} {rate:>9.1f} {w:>9.3f} {vend:>10.1f}")
            print("-" * 46)
    print("\n判读：")
    print("· T1.4：基准档(V12=13, τ×1)各项 ±10mV/倍乘后方向不变（rate 单调、无崩）即过；")
    print("· T1.5：τ× 扫描若 rate 呈中间凸（如 ×0.5–1 最高、两端低）→ 记录最优点；")
    print("· g_Kv3=0 臂应对 V12/τ 扫描不敏感（平线），作机制对照。")


if __name__ == "__main__":
    main()