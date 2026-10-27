"""T1.2b：Na 慢失活（主电流乘 s 形式）——追求"完全阻滞"。

实现（2026-10-10）：I_Na = g_Na·m³·h·s（s：τ_s∈[50,300]ms，s∞ 静息≈0.9、
持续去极化平台渐关）。完成原型 docstring 留下的 TODO（Erisir/Fleidervish 型）。
注意：主 Na 乘 s 使基线钠流降 ~10%，gKv3=40 组 rate 会略低于 T1.1 对拍值，
判读以"是否出现完全阻滞"为准，速率绝对值以本表为准重新记录。

网格：g_Kv3 {0,40} × τ_s {50,100,200,300}ms × I {45,60} μA/cm²（16 次运行）
成功判据：gKv3=0 组某档 rate <5Hz（完全阻滞），gKv3=40 组全存活。

用法（mleDSar 环境）：
    python stage1_single_cell/run_t12b.py
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


def one_run(g_kv3, tau_s_ms, i_uA_cm2):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(0)
    G = make_pv_neuron(1, reset=None, refractory="v > -20.0 * mV")
    G.g_Kv3 = g_kv3 * (mS / cm2)
    G.tau_s_na = tau_s_ms
    G.I_inj = i_uA_cm2 * (uA / cm2)
    mon = StateMonitor(G, "v", record=True)
    sp = SpikeMonitor(G)
    run(DUR_MS * ms, report=None)
    rate = len(sp.t) / (DUR_MS / 1000.0)
    w, vend = measure(np.array(mon.v[0] / mV), DT_MS, rate)
    return rate, w, vend


def main():
    print("T1.2b：Na 慢失活完全阻滞扫描（主电流×s 形式）")
    print(f"{'gKv3':>5} {'tau_s':>6} {'I':>4} {'rate(Hz)':>9} {'half(ms)':>9} "
          f"{'v_end(mV)':>10} {'判定':>6}")
    print("-" * 58)
    ok_block = False
    ok_survive = True
    for g_kv3 in [0.0, 40.0]:
        for tau_s in [50.0, 100.0, 200.0, 300.0]:
            for i_inj in [45.0, 60.0]:
                rate, w, vend = one_run(g_kv3, tau_s, i_inj)
                is_block = rate < 5.0
                if g_kv3 == 0.0 and is_block:
                    ok_block = True
                if g_kv3 == 40.0 and rate < 100.0:
                    ok_survive = False
                print(f"{g_kv3:>5.1f} {tau_s:>6.0f} {i_inj:>4.0f} {rate:>9.1f} "
                      f"{w:>9.3f} {vend:>10.1f} {'BLOCK' if is_block else 'OK':>6}")
            print("-" * 58)
    print()
    print(f"判据汇总：无Kv3 组完全阻滞出现 = {'✅' if ok_block else '❌'}；"
          f"有Kv3 组全部存活 = {'✅' if ok_survive else '❌'}")
    if ok_block and ok_survive:
        print("→ T1.2 完全阻滞判据达成；选定档回填 params（tau_s_na）+ 沿革。")
    elif not ok_block:
        print("→ 未完全阻滞：τ_s 加长（→500/1000ms）或 DUR 延至 5s；仍不崩则"
              "查 s_inf 中点（−45 再左移会削弱静息可用度）。")


if __name__ == "__main__":
    main()