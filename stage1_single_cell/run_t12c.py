"""T1.2 攻坚第二轮：τ_p 扫描——验证"τ_p 过慢致 p 不激活"因果链。

背景（run_t12b 结论）：p 峰仅 0.034 → I_Kv3 比 I_Kdr 小 10 倍 → Kv3 无效；
尖峰宽度由 φ=5 加速的 h（钠自终止）决定，钾侧旋钮无效。
本脚本：tau_p_slow ∈ {0, 0.5, 1.0, 2.0}（0 = 常数 τ_p=0.15ms，路线图"≥0.15ms"下界档）
× g_Kv3{0,20} @ g_Kdr=6, I=6；成功判据：半宽对比 ≥2× 且 rate ≥200Hz（不破坏 T1.1）。

用法（mleDSar 环境）：
    python stage1_single_cell/run_t12c.py
定档后：回填 params_v1.4.md 沿革 + 原型脚本比对（推翻条件：原型 τ_p 形式不同）。
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

DUR_MS = 2000.0
DT_MS = 0.025
E_K = PARAMS["cell"].E_K


def run_one(tau_p_slow, g_kv3, i_uA_cm2=6.0):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(0)
    # reset=None + 条件不应期（原型 spike 语义；同 run_t12b）
    G = make_pv_neuron(1, reset=None, refractory="v > -20.0 * mV")
    G.tau_p_slow = tau_p_slow * ms
    G.g_Kv3 = g_kv3 * (mS / cm2)
    G.I_inj = i_uA_cm2 * (uA / cm2)
    mon = StateMonitor(G, "v", record=True)
    gmon = StateMonitor(G, ["p", "q"], record=True) if g_kv3 > 0 else None
    sp = SpikeMonitor(G)
    run(DUR_MS * ms, report=None)
    v = np.array(mon.v[0] / mV)
    rate = len(sp.t) / (DUR_MS / 1000.0)
    w, vend = measure(v, DT_MS, rate)
    extra = {}
    if gmon is not None:
        p = np.array(gmon.p[0])
        q = np.array(gmon.q[0])
        i_kv3 = g_kv3 * p ** 2 * q * (E_K - v)
        extra = {"p_max": p.max(), "i_kv3_peak": np.abs(i_kv3).max()}
    return rate, w, vend, extra


def main():
    print("τ_p 扫描 @ g_Kdr=6, I=6（τ_p_slow=0 即常数 τ_p=0.15ms）")
    print(f"{'τpslow':>7} {'gKv3':>5} {'rate(Hz)':>9} {'half(ms)':>9} "
          f"{'p_max':>7} {'I_Kv3pk':>8} {'v_end(mV)':>10}")
    print("-" * 64)
    res = {}
    for tps in [0.0, 0.5, 1.0, 2.0]:
        for g_kv3 in [0.0, 20.0]:
            rate, w, vend, ex = run_one(tps, g_kv3)
            res[(tps, g_kv3)] = (rate, w)
            pmax = f"{ex['p_max']:.3f}" if ex else "-"
            ipk = f"{ex['i_kv3_peak']:.1f}" if ex else "-"
            print(f"{tps:>7.2f} {g_kv3:>5.1f} {rate:>9.1f} {w:>9.3f} "
                  f"{pmax:>7} {ipk:>8} {vend:>10.1f}")
        h0, h20 = res[(tps, 0.0)][1], res[(tps, 20.0)][1]
        ok = (not np.isnan(h0) and not np.isnan(h20) and h20 > 0
              and h0 / h20 >= 2 and res[(tps, 20.0)][0] >= 200)
        tag = "✓ 对比显现+频率达标" if ok else "✗"
        print(f"  → 半宽对比 = {h0/h20:.2f}×（无/有 Kv3）  {tag}")
        print("-" * 64)
    print("判读：τ_p_slow=0 若显现对比 → 采常数 τ_p=0.15ms 档，")
    print("回填 params（tau_p_slow=0.0）+ 沿革记录 + T1.4 敏感性顺延；")
    print("仍不显 → 等原型脚本比对，转 Erisir 两态 Kv3 方程预案。")


if __name__ == "__main__":
    main()