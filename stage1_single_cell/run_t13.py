"""T1.3：g_Kv3 × I 的 f-I 曲面 + 半宽 + 末段电位（Fig 1 右半素材）。

网格：g_Kv3 ∈ {0,5,10,15,20,30,40} mS/cm² × I ∈ 0–60 μA/cm²（步长 5，原型 0.5nA 当量）
共 91 次单细胞运行（每次 2s 生物时间）。g_Na_slow=0（T1.2b 重写前的基线曲面；
g_Kv3=0 行的 BLOCK 区暂按"软失活"判据标定，T1.2b 后升级）。

输出：
  - 终端速率表 / 末段电位表 / 半宽表（紧凑打印）
  - stage1_single_cell/t13_fI_surface.csv（机读）
  - figures/f1_fI_surface.png（双面板：f-I 曲线族 + v_end 曲线族，BLOCK 点红叉）

用法（mleDSar 环境）：
    python stage1_single_cell/run_t13.py
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

G_KV3_LIST = [0, 5, 10, 15, 20, 30, 40]
I_LIST = list(range(0, 85, 5))     # μA/cm²（0.5nA 步长当量）
DUR_MS = 2000.0
DT_MS = 0.025


def blocked(rate, v_end):
    """软失活/阻滞判据（T1.2b 后口径）。

    注：I=0 静默不算 BLOCK（rate<5 但 v_end 就在静息电位附近），
    仅当既有放电近停、末端又持续去极化才标。
    """
    if rate < 5.0 and v_end < -55.0:   # 静默且 v_end 接近静息 → 不是 BLOCK
        return False
    return rate < 5.0 or v_end > -35.0


def one_run(g_kv3, i_uA_cm2):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(0)
    G = make_pv_neuron(1, reset=None, refractory="v > -20.0 * mV")
    G.g_Kv3 = g_kv3 * (mS / cm2)
    G.I_inj = i_uA_cm2 * (uA / cm2)
    mon = StateMonitor(G, "v", record=True)
    sp = SpikeMonitor(G)
    run(DUR_MS * ms, report=None)
    rate = len(sp.t) / (DUR_MS / 1000.0)
    w, vend = measure(np.array(mon.v[0] / mV), DT_MS, rate)
    return rate, w, vend


def main():
    n_i = len(I_LIST)
    rates = np.zeros((len(G_KV3_LIST), n_i))
    vends = np.zeros_like(rates)
    halves = np.full_like(rates, np.nan)
    print(f"g_Kv3 扫描 {G_KV3_LIST} × I {I_LIST[0]}–{I_LIST[-1]} μA/cm²，共 "
          f"{len(G_KV3_LIST) * n_i} 次运行…")
    for r, g in enumerate(G_KV3_LIST):
        for c_i, i_inj in enumerate(I_LIST):
            rate, w, vend = one_run(g, i_inj)
            rates[r, c_i] = rate
            vends[r, c_i] = vend
            halves[r, c_i] = w
        print(f"  g_Kv3={g:>2} 完成（I=60 档: rate={rates[r, -1]:6.1f}Hz "
              f"v_end={vends[r, -1]:6.1f}mV）")

    # ---- 终端表 ----
    print("\n=== rate (Hz) ===")
    print(f"{'gKv3|I':>7}" + "".join(f"{i:>7}" for i in I_LIST))
    for r, g in enumerate(G_KV3_LIST):
        print(f"{g:>7}" + "".join(f"{rates[r, c]:>7.0f}" for c in range(n_i)))
    print("\n=== v_end (mV；* = BLOCK 判据触发) ===")
    print(f"{'gKv3|I':>7}" + "".join(f"{i:>7}" for i in I_LIST))
    for r, g in enumerate(G_KV3_LIST):
        row = ""
        for c in range(n_i):
            mark = "*" if blocked(rates[r, c], vends[r, c]) else ""
            row += f"{vends[r, c]:>6.0f}{mark[:1] or ' '}"
        print(f"{g:>7}{row}")
    print("\n=== half-width (ms) ===")
    print(f"{'gKv3|I':>7}" + "".join(f"{i:>7}" for i in I_LIST))
    for r, g in enumerate(G_KV3_LIST):
        print(f"{g:>7}" + "".join(
            f"{halves[r, c]:>7.2f}" if not np.isnan(halves[r, c]) else f"{'-':>7}"
            for c in range(n_i)))

    # ---- CSV ----
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "t13_fI_surface.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("g_kv3_mScm2,i_uA_cm2,rate_hz,half_ms,v_end_mV,blocked\n")
        for r, g in enumerate(G_KV3_LIST):
            for c, i_inj in enumerate(I_LIST):
                f.write(f"{g},{i_inj},{rates[r, c]:.1f},"
                        f"{halves[r, c] if not np.isnan(halves[r, c]) else ''},"
                        f"{vends[r, c]:.1f},"
                        f"{int(blocked(rates[r, c], vends[r, c]))}\n")
    print(f"\nCSV 已保存: {csv_path}")

    # ---- Fig 1 候选图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(G_KV3_LIST)))
    for r, g in enumerate(G_KV3_LIST):
        lbl = f"g_Kv3={g}"
        axes[0].plot(I_LIST, rates[r], "-o", color=cmap[r], ms=3, label=lbl)
        axes[1].plot(I_LIST, vends[r], "-o", color=cmap[r], ms=3, label=lbl)
        for c in range(len(I_LIST)):
            if blocked(rates[r, c], vends[r, c]):
                axes[0].plot(I_LIST[c], rates[r, c], "x", color="red", ms=7)
                axes[1].plot(I_LIST[c], vends[r, c], "x", color="red", ms=7)
    axes[0].set(xlabel="I (μA/cm²)", ylabel="rate (Hz)", title="f-I 曲面")
    axes[1].axhline(-35, color="red", ls=":", lw=1)
    axes[1].set(xlabel="I (μA/cm²)", ylabel="v_end (mV)", title="末段电位（红点=BLOCK）")
    axes[1].legend(fontsize=7, ncol=2)
    out = os.path.join(fig_dir, "f1_fI_surface.png")
    fig.savefig(out, dpi=150)
    print(f"图已保存: {out}")
    print("\n判读：g_Kv3=0 行高 I 端应出现 BLOCK 标记；rate 随 g_Kv3 增大单调下移；"
          "v_end 随 g_Kv3 增大深复极。")


if __name__ == "__main__":
    main()
