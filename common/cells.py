"""Neuron equations: Kv3-PV 快速放电中间神经元（原型血统回归版）。

血统（2026-10-10 裁定，经 WB96 原文逐字核对）：以 2026-09-22 原型《Kv3_PV_Brian2起始脚本》为基准——
  - Na/Kdr 门控 = **Traub-Miles 型**（Vr=−60mV；Destexhe 系皮层模型惯例）。
    ⚠️ 勿标注"Wang-Buzsáki 门控"：WB96 原文实为经典 HH 平移型（α 用 V+35/+34/+58）、
    g_Na=35、g_K=9、且原文自带 φ=5——与原型不同源。⚠️ 勿换回教科书 HH 型
    （2026-10-10 前 repo 版本因此全盘失效）。两血统的差异与出处入 params_v1.4.md 沿革。
  - 电导：g_Na=100 / g_Kdr=20 / g_Kv3=40 mS/cm²（原型值）
  - 无 φ 温度系数（原型无；phi 保留为参数默认 1.0；WB96 原文 φ=5 留作 T1.4 敏感性）
  - Kv3 门控：p(V1/2=+13,k=12; τ_p=0.15+2/(1+exp((V+10)/15))ms)、
    q(q∞=0.15+0.85/(1+exp((V+16)/12)); τ_q=300ms)——与原型逐字一致
  - spike 语义：threshold="v > -20mV"，refractory="v > -20mV"（条件式：
    尖峰降到 −20 以下才允许再次触发），reset=None（自然复极，v 不被钳制）

I_gap / I_syn 为 summed 突触电流接收变量（网络用）；I_Na_slow（s 门控）T1.2 开关。

换算备忘：原型 area=10000μm²、注入 pA → 1nA ≡ 10μA/cm²（"6nA" = 60μA/cm²）。
"""
from brian2 import NeuronGroup, Equations, ms, mV, mS, cm, uF, uA

from .params import PARAMS, CellParams


def make_pv_neuron(N: int, params: CellParams = None, *,
                   reset=None, refractory="v > -20.0 * mV"):
    p = params or PARAMS["cell"]
    eqs = Equations("""
        dv/dt = (I_leak + I_Na + I_Kdr + I_Kv3 + I_Na_slow + I_gap + I_syn + I_inj + I_GABA_B) / C : volt
        dh/dt = phi * (alpha_h * (1 - h) - beta_h * h) : 1
        dn/dt = phi * (alpha_n * (1 - n) - beta_n * n) : 1
        dm/dt = phi * (alpha_m * (1 - m) - beta_m * m) : 1
        dp/dt = (p_inf - p) / tau_p : 1
        dq/dt = (q_inf - q) / tau_q : 1
        ds/dt = (s_inf - s) / tau_s : 1

        I_leak   = g_leak * (E_leak - v) : amp/meter**2
        # 慢失活乘在主 Na 电流上（T1.2b 2026-10-10：静息 s≈0.9，持续去极化平台渐关 → 阻滞）
        I_Na     = g_Na * m**3 * h * s * (E_Na - v) : amp/meter**2
        I_Kdr    = g_Kdr * n**4 * (E_K - v) : amp/meter**2
        I_Kv3    = g_Kv3 * p**2 * q * (E_K - v) : amp/meter**2
        I_Na_slow = g_Na_slow * m**3 * h * s * (E_Na - v) : amp/meter**2
        I_gap    : amp/meter**2
        I_syn    : amp/meter**2
        I_inj    : amp/meter**2
	I_GABA_B : amp/meter**2

        # --- Na / Kdr：Traub-Miles 型（原型；Vr=−60mV） ---
        alpha_m = 0.32/mV/ms * (13*mV - v + Vr) / (exp((13*mV - v + Vr)/(4*mV)) - 1) : Hz
        beta_m  = 0.28/mV/ms * (v - Vr - 40*mV) / (exp((v - Vr - 40*mV)/(5*mV)) - 1) : Hz
        alpha_h = 0.128/ms * exp((17*mV - v + Vr)/(18*mV)) : Hz
        beta_h  = 4/ms / (1 + exp((40*mV - v + Vr)/(5*mV))) : Hz
        alpha_n = 0.032/mV/ms * (15*mV - v + Vr) / (exp((15*mV - v + Vr)/(5*mV)) - 1) : Hz
        beta_n  = 0.5/ms * exp((10*mV - v + Vr)/(40*mV)) : Hz

        # --- Kv3：激活 p（高阈、快），慢失活 q ---
        p_inf = 1 / (1 + exp(-(v - V12_p) / k_p)) : 1
        tau_p = tau_p_min + tau_p_slow / (1 + exp((v + 10*mV) / (15*mV))) : second
        q_inf = q_inf_a + (1 - q_inf_a) / (1 + exp((v - q_inf_half) / q_inf_k)) : 1

        # --- Na 慢失活（T1.2 开关；g_Na_slow 默认 0） ---
        s_inf = 1 / (1 + exp((v - s_inf_half) / s_inf_k)) : 1
        tau_s = tau_s_na * ms : second

        C : farad/meter**2
        g_leak : siemens/meter**2
        E_leak : volt
        g_Na : siemens/meter**2
        E_Na : volt
        g_Kdr : siemens/meter**2
        E_K : volt
        g_Kv3 : siemens/meter**2
        V12_p : volt
        k_p : volt
        Vr : volt
        tau_p_min : second
        tau_p_slow : second
        tau_q : second
        q_inf_a : 1
        q_inf_half : volt
        q_inf_k : volt
        phi : 1
        s_inf_half : volt
        s_inf_k : volt
        tau_s_na : 1
        g_Na_slow : siemens/meter**2
    """)

    kwargs = dict(threshold="v > -20.0 * mV", method="rk4", dt=0.025 * ms)
    if reset is not None:
        kwargs["reset"] = reset
    if refractory is not None:
        kwargs["refractory"] = refractory
    G = NeuronGroup(N, eqs, **kwargs)

    G.C = p.C * uF / cm**2
    G.g_leak = p.g_leak * mS / cm**2
    G.E_leak = p.E_leak * mV
    G.g_Na = p.g_Na * mS / cm**2
    G.E_Na = p.E_Na * mV
    G.g_Kdr = p.g_Kdr * mS / cm**2
    G.E_K = p.E_K * mV
    G.g_Kv3 = p.g_Kv3 * mS / cm**2
    G.V12_p = p.V12_p * mV
    G.k_p = p.k_p * mV
    G.Vr = p.Vr * mV
    G.tau_p_min = p.tau_p_min * ms
    G.tau_p_slow = p.tau_p_slow * ms
    G.tau_q = p.tau_q * ms
    G.q_inf_a = p.q_inf_a
    G.q_inf_half = p.q_inf_half * mV
    G.q_inf_k = p.q_inf_k * mV
    G.phi = p.phi
    G.s_inf_half = p.s_inf_half * mV
    G.s_inf_k = p.s_inf_k * mV
    G.tau_s_na = p.tau_s_na
    G.g_Na_slow = 0.0 * mS / cm**2  # T1.2 开启（τ_s 扫描走 tau_s_range）
    G.I_gap = 0.0 * uA / cm**2
    G.I_syn = 0.0 * uA / cm**2
    G.I_inj = 0.0 * uA / cm**2
    G.I_GABA_B = 0.0 * uA / cm**2

    G.v = -65 * mV
    G.h = 0.6
    G.n = 0.3
    G.m = 0.05
    G.p = 0.0
    G.q = 0.9
    G.s = 1.0

    return G


# 阶段 1 验收锚（原型口径）：f-I >200Hz@强驱动；有无 Kv3 半宽对比 ≥2×（Erisir 药理学锚）；
# 强驱动末段维持（原型：含 Kv3 −49mV vs 无 Kv3 −34mV 失活区）
def acceptance_check(spike_trains, v_traces, i_inj):
    """Stage 1 acceptance (T1.1/T1.2 placeholder)。"""
    return []
