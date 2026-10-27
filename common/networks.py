"""网络构建器：ING（主）+ PING（对照）。

T2.1a 更新（2026-10-17）：make_ing 换动力学 GABA_A，逐字移植 benchmark 已验证实现
（run_benchmark_brian2.py；T2.0 f0=46.3Hz / T2.1 诊断 22.4Hz 均出自该方程组）。
- GABA_A：dg/dt = -g/τ（τ=10ms），电导型 + 饱和 clip(g,0,1)，
  逐边归一化 g_syn_each = G_GABA_TOTAL/(N*p_II) → 每细胞平均总抑制 ≈ G_GABA_TOTAL
- gap：逐边归一化 g_gap_each = g_gap_default/(N*p_GAP)（v3.1-fix2 量纲修正版）
- spike 语义：reset=None + refractory="v > -20.0*mV"（血统纪律，不动）
模块级常量 G_GABA_TOTAL / E_GABA / P_GAP 为 T2.1a 占位，T2.1b 迁入 params.py。
"""
from brian2 import NeuronGroup, Synapses, ms, mV, mS, cm2, uA, clip
import numpy as np

from .cells import make_pv_neuron
from .params import PARAMS

# GABA_A 目标量（T2.1b 迁入 params.NetParams）
G_GABA_TOTAL = 0.1      # mS/cm² 每细胞平均总抑制
E_GABA = -75.0          # mV
P_GAP = 0.1             # gap 连接概率（与 p_connect_II 并列）


def make_ing(N: int = None, gap_junction: bool = True,
             i_mu: float = 1.0, seed: int = 42,
             g_gaba_total: float = None):
    """ING 网络：N 个 PV 中间神经元 + 动力学 GABA_A + 可选 gap junction。

    N            : 神经元数（默认 params.net.N_PV）
    gap_junction : 是否加电突触
    i_mu         : 平均驱动 μA/cm²（I_hetero 高斯 hetero，负值截断 0）
    seed         : hetero 抽样种子（brian2 全局 seed 由调用方负责）
    g_gaba_total : mS/cm² 总抑制（默认模块常量 G_GABA_TOTAL；T2.1b 扫描轴）
    """
    p = PARAMS["net"]
    N = N or p.N_PV
    G = make_pv_neuron(N, reset=None, refractory="v > -20.0 * mV")

    rng = np.random.default_rng(seed)
    i_app = i_mu * (1.0 + p.I_hetero * rng.standard_normal(N))
    i_app = np.clip(i_app, 0.0, None)
    G.I_inj = i_app * (uA / cm2)

    # --- GABA_A：动力学衰减 + 电导型饱和（benchmark 同构） ---
    P_II = p.p_connect_II
    ggt = g_gaba_total if g_gaba_total is not None else G_GABA_TOTAL
    g_syn_each = ggt / (N * P_II)
    syn_ii = Synapses(G, G,
                      model=f"""
        dg/dt = -g / ({p.tau_syn_GABA_A} * ms) : 1 (clock-driven)
        I_syn_post = g_syn_each * g * ({E_GABA} * mV - v_post) : amp/meter**2 (summed)
        g_syn_each : siemens/meter**2 (constant)
        """,
                      on_pre="""g += 1
        g = clip(g, 0, 1)""",
                      method="exact")
    syn_ii.connect(p=P_II)
    syn_ii.g_syn_each = g_syn_each * mS / cm2

    # --- gap junction：连续求和电流（v3.1-fix2），逐边归一化 ---
    gap = None
    if gap_junction:
        g_gap_each = p.g_gap_default / (N * P_GAP)
        gap = Synapses(G, G,
                       model="""
        I_gap_post = g_gap * (v_pre - v_post) : amp/meter**2 (summed)
        g_gap : siemens/meter**2 (constant)
        """)
        gap.connect(p=P_GAP)
        gap.g_gap = g_gap_each * mS / cm2

    return G, syn_ii, gap


def make_ping(N_pv: int = None, N_pyr: int = None):
    """PING 网络：PV + 锥体（AMPA E→I、GABA_A/B I→E）。阶段 2 后段实现。"""
    raise NotImplementedError("PING 对照网络于阶段 2 后段实现（见路线图 4.1）")