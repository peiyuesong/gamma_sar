"""T2.4a: GABA_B 单细胞门控验收（最终修正版 v7）。

关键修正：
1. K1/K2/K3_eff/K4 全部乘以 1000（params.py 是基于 ms 的单位，方程默认 second）
2. 用 TimedArray 生成 2s 持续递质脉冲，避免 int() 布尔解析问题
3. 验收：IPSP >= 5mV 且衰减时程 >= 100ms
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, defaultclock, seed, ms, mV, mS, cm,
                    uF, mM, Hz, TimedArray, StateMonitor, Network,
                    NeuronGroup, Equations)
from common.params import PARAMS

p_gaba = PARAMS["gabab"]

C_M = 1.0
G_LEAK = 0.1
E_LEAK = -65.0
E_K = -90.0

T_ON_MS = 100.0
T_DUR_MS = 2000.0
T_OFF_MS = T_ON_MS + T_DUR_MS
T_TOTAL_MS = T_OFF_MS + 1000.0


def main():
    start_scope()
    defaultclock.dt = 0.025 * ms
    seed(42)

    n = p_gaba.n_coop
    K3_eff = p_gaba.K3 / (p_gaba.K_d ** (1.0 / n))

    # 用 TimedArray 生成 T(t)：1ms 分辨率
    n_pts = int(T_TOTAL_MS) + 1
    t_pts = np.arange(n_pts)
    T_vals = np.zeros(n_pts)
    T_vals[int(T_ON_MS):int(T_OFF_MS)] = 1.0  # mM
    T_stim = TimedArray(T_vals * mM, dt=1 * ms)

    eqs = Equations("""
        dv/dt = (I_leak + I_GABA_B) / C : volt
        I_leak = g_leak * (E_leak - v) : amp/meter**2

        dr/dt = K1 * T * (1 - r) - K2 * r : 1
        ds/dt = K3_eff * r - K4 * s : 1

        T = T_stim(t) : mM

        I_GABA_B = g_GABA_B * (s**n / (s**n + 1)) * (E_K - v) : amp/meter**2

        C : farad/meter**2
        g_leak : siemens/meter**2
        E_leak : volt
        E_K : volt
        g_GABA_B : siemens/meter**2
        K1 : Hz/mM
        K2 : Hz
        K3_eff : Hz
        K4 : Hz
        n : 1
    """)

    PV = NeuronGroup(1, eqs, method="rk4", namespace={"T_stim": T_stim})

    PV.C = C_M * (uF / (cm**2))
    PV.g_leak = G_LEAK * (mS / (cm**2))
    PV.E_leak = E_LEAK * mV
    PV.E_K = E_K * mV

    # 关键：乘以 1000 转成 1/s 单位
    PV.K1 = p_gaba.K1 * 1000 * (Hz / mM)
    PV.K2 = p_gaba.K2 * 1000 * Hz
    PV.K3_eff = K3_eff * 1000 * Hz
    PV.K4 = p_gaba.K4 * 1000 * Hz
    PV.n = n

    PV.g_GABA_B = p_gaba.g_GABA_B * (mS / (cm**2))

    PV.v = E_LEAK * mV
    PV.r = 0
    PV.s = 0

    mon = StateMonitor(PV, ["v", "s", "r"], record=True)
    net = Network(PV, mon)
    net.run(T_TOTAL_MS * ms)

    v_arr = np.array(mon.v[0] / mV)
    t_arr = np.array(mon.t / ms)
    s_arr = np.array(mon.s[0])
    r_arr = np.array(mon.r[0])

    v_rest = E_LEAK
    v_min = float(v_arr.min())
    ipsp_amp = v_rest - v_min

    threshold = v_rest - 2.0
    post_off = t_arr >= T_OFF_MS
    recovered = post_off & (v_arr >= threshold)
    if recovered.any():
        t_recover = float(t_arr[recovered].min())
        duration = t_recover - T_OFF_MS
    else:
        duration = float(T_TOTAL_MS - T_OFF_MS)

    print("=" * 55)
    print("T2.4a GABA_B 单细胞门控验收 (2s 持续激活)")
    print("=" * 55)
    print(f"  K1={PV.K1}, K2={PV.K2}")
    print(f"  K3_eff={PV.K3_eff}, K4={PV.K4}")
    print(f"  g_GABA_B={p_gaba.g_GABA_B}, n={n}, K_d={p_gaba.K_d}")
    print(f"  静息 = {v_rest:.1f} mV | 最低 = {v_min:.1f} mV")
    print(f"  IPSP 幅度 = {ipsp_amp:.2f} mV  (>= 5mV ?)")
    print(f"  衰减时程 = {duration:.1f} ms  (>= 100ms ?)")
    verdict = ("PASS" if ipsp_amp >= 5.0 and duration >= 100 else
               "DUR_ADJUST" if ipsp_amp >= 5.0 else "FAIL")
    print(f"  verdict = {verdict}")
    print("=" * 55)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(t_arr, v_arr, 'b-', lw=1.0)
    ax1.axvspan(T_ON_MS, T_OFF_MS, color='r', alpha=0.1)
    ax1.axhline(v_rest, color='gray', ls=':')
    ax1.set_ylabel("V (mV)")
    ax1.set_title(f"IPSP={ipsp_amp:.1f}mV, dur={duration:.0f}ms [{verdict}]")

    ax2.plot(t_arr, r_arr, 'orange', lw=1.0)
    ax2.axvspan(T_ON_MS, T_OFF_MS, color='r', alpha=0.1)
    ax2.set_ylabel("r (1)")

    ax3.plot(t_arr, s_arr, 'g-', lw=1.0)
    ax3.axvspan(T_ON_MS, T_OFF_MS, color='r', alpha=0.1)
    ax3.set_ylabel("s (norm)")
    ax3.set_xlabel("Time (ms)")

    plt.tight_layout()
    fig_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "f_t24a_gabab_ipsp.png"
    )
    plt.savefig(fig_path, dpi=150)
    print(f"saved: {fig_path}")


if __name__ == "__main__":
    main()