"""全部参数集中于此（单一事实源）。

版本锚定：v3.1-fix2（2026-09-28，Glinski 2025 原文到位修订）。
fix2 变更：
- iTBS 簇参数按 Glinski 2025 正文+补充材料回填（原 iti=8.0 单位/语义错误：
  8 是秒级 inter-cycle，不是 ms 簇间间隔）
- CellParams 新增 tau_p_slow（原型 tau_p 形式的慢分量 2.0ms）
"""
from dataclasses import dataclass, field
from typing import List


# ---------- 单神经元（Kv3 通道 + Na 慢失活） ----------
@dataclass
class CellParams:
    # Wang-Buzsáki 框架基础（不改）
    C: float = 1.0           # μF/cm²
    g_leak: float = 0.1      # mS/cm²
    E_leak: float = -65.0    # mV
    g_Na: float = 100.0      # mS/cm²（原型值；2026-10-10 血统裁定回归）
    E_Na: float = 55.0       # mV
    g_Kdr: float = 20.0      # mS/cm²（原型值）
    E_K: float = -90.0       # mV（同时为 GABA_B 反转电位）

    # Kv3 通道（v3.1-fix：按 Rudy & McBain 2001 克隆值；天然 Lien & Jonas 2003 −12.4mV 入敏感性扫描）
    g_Kv3: float = 40.0      # mS/cm²（原型值；扫描轴 1）
    V12_p: float = 13.0      # mV
    k_p: float = 12.0        # mV
    Vr: float = -60.0        # mV（Traub-Miles 门控参考电位，原型值）
    tau_p_min: float = 0.15  # ms（原型形式 τ_p = 0.15 + 2/(1+exp((V+10)/15))）
    tau_p_slow: float = 2.0  # ms（fix2 新增，原型慢分量）
    q_inf_a: float = 0.15
    q_inf_half: float = -16.0  # mV（q∞=0.15+0.85/(1+exp((V+16)/12))）
    q_inf_k: float = 12.0
    tau_q: float = 300.0     # ms（型号待核；任务 0.2 查 Joho 1999 真出处）
    phi: float = 1.0         # 门控速率倍乘；原型无温度系数（2026-10-10 裁定）；T1.4 敏感性用

    # Na 慢失活（Erisir/Fleidervish 型；T1.2 开启）
    g_Na_slow: float = 0.0   # 待阶段 1 T1.2 标定
    tau_s_range: tuple = (50.0, 300.0)  # ms（T1.2 扫描轴）
    s_inf_half: float = -45.0  # mV（T1.2b 功能规格：静息可用≈0.90、−45 半失活、平台期渐关；Fleidervish 型声明）
    s_inf_k: float = 7.0       # mV
    tau_s_na: float = 100.0    # ms（基准档；扫描用 tau_s_range）


# ---------- GABA_B（v3.1-fix：Destexhe 1998 原文常数） ----------
@dataclass
class GABABParams:
    """Destexhe 1998 简约式 + Thomson & Destexhe 1999 实验背书。

    注意：原文 K4=34s⁻¹（τ≈29ms）；本文档用 τ_s=130ms 作扫描选择，
    IPSP 100–200ms 由 n=4 协同门控获得。
    """
    K1: float = 0.09         # ms⁻¹(mM)⁻¹（=9×10⁴ M⁻¹s⁻¹）
    K2: float = 0.0012       # ms⁻¹（=1.2 s⁻¹）
    K3: float = 0.18         # μM·ms⁻¹（=180 s⁻¹）
    K4: float = 1.0 / 130.0  # ms⁻¹（τ_s=130ms；非 Destexhe 原文 34s⁻¹）
    tau_s_scan: tuple = (100.0, 300.0)  # ms
    g_GABA_B: float = 1.0    # mS/cm²（占位；T2.4b 反标定）
    K_d: float = 100.0       # μM⁴（s⁴/(s⁴+K_d) 形式原常数）
    n_coop: int = 4          # 协同位点数（Thomson & Destexhe 1999 实验 4 结合位点）


# ---------- 网络 ----------
@dataclass
class NetworkParams:
    N_PV: int = 200                 # ING 主版
    N_Pyr: int = 800                # PING 对照
    p_connect_EE: float = 0.05
    p_connect_EI: float = 0.2
    p_connect_II: float = 0.2
    tau_syn_AMPA: float = 2.0       # ms
    tau_syn_GABA_A: float = 10.0    # ms
    g_gap_range: tuple = (0.0, 1.0)  # mS（扫描轴 2；线性近似，整流性声明入补充材料）
    g_gap_default: float = 0.2       # mS/cm² 基准档（占位；T2.1 标定密度折算）
    I_hetero: float = 0.10          # ±10% 异质性


# ---------- 刺激 ----------
@dataclass
class StimParams:
    # tACS
    kappa: float = 0.25             # mV per (mV/mm)
    kappa_scan: tuple = (0.1, 0.5)
    E0_scan: tuple = (0.2, 1.0)     # V/m（Glinski 原文无电场估计，按同类 1mA 文献惯例；fix2）
    freq_scan: tuple = (30, 80)     # Hz
    freq_step: float = 5.0

    # iTBS（Glinski 2025 修改版，fix2 已提取回填）
    burst_freq: float = 5.0         # Hz 簇频率（2s 开/8s 关）
    n_pulses_per_burst: int = 3     # 脉冲/簇
    intraburst_freq: float = 40.0   # Hz 簇内频率（25ms 间隔；经典 50Hz 改为 40Hz）
    inter_burst_ms: float = 200.0   # 簇间间隔（5Hz）
    inter_cycle_s: float = 8.0      # 每块 190s 后的间歇（原 iti=8.0 单位错误，fix2 更正）
    block_s: float = 190.0          # 每块时长（600 脉冲）
    n_blocks: int = 4               # @0/5/10/15min
    total_pulses: int = 2400        # 每侧
    intensity_pct_amt: float = 80.0 # %AMT

    # 光
    retino_thal_delay: tuple = (30.0, 80.0)  # ms

    # 振动
    meissner_tune: float = 40.0     # Hz（任务 0.2 落文献）

    # PBM（预测级）
    slow_mod_tau_range: tuple = (60.0, 600.0)  # s

    # 状态操纵
    mu_EO: float = 1.0              # 基准驱动
    mu_EC: float = 1.0 / 1.8        # 冻结；±10% 微调，禁止出此区间


# ---------- 相位门控 STDP ----------
@dataclass
class STDParams:
    a_amp_range: tuple = (0.2, 0.8)        # G(φ) 振幅扫描
    phi_peak: float = 90.0                  # 度；H1 假设，建议 held-out 验证
    tau_trace_set: tuple = (30, 60, 120)    # min


# ---------- P2 验收阈值（fix2 已回填：Glinski 2025 提取） ----------
P2_THRESHOLD_FACTOR = 0.8          # 维持幅度 ≥ 实测(vs-sham +31%) × 0.8 = +25%
P2_THRESHOLD_ABS = 0.25            # 模型 EO 2h 维持（vs 自身 sham）下限


@dataclass(frozen=True)
class AcceptanceParams:
    """P1–P4 验收判据（任务 0.1 数据卡回填值，口径见 data/glinski2025_data_card.md）"""
    p2_maintain_min: float = 0.25   # EO 2h 维持（vs 自身 sham）≥ +25%
    # 推导：实测 X-tACS Peak EO 90–120min vs-sham ≈ +31% × 0.8 = 24.8%（图读 ±0.02）
    # 备口径（仅图注参考，不作判据）：vs-baseline +48% × 0.8 = +38.4%
    p3_ec_late_trough_min: float = -0.20   # EC 90–120min Trough −0.20（p<0.0001）
    p3_ec_late_peak: float = -0.14         # EC 90–120min Peak −0.14（p=0.004）
    p3_ec_late_tacs_ns: float = 0.00       # tACS ≈ sham（p=0.92）；iTBS −0.03（p=0.63）ns
    p4_f0_window_hz: float = 10.0          # 响应峰须在 f0 ± 10Hz（硬判据）
    # P1：EO 排序 Peak>tACS≈iTBS>Trough>sham（Wilcoxon + BH-FDR，10 seeds）
    # P3 辅约束（EC 30–60min）：tACS +0.11（p=0.03）、Trough −0.10（p=0.03）


# 全局聚合（导入此变量即可访问全部参数）
PARAMS = {
    "cell": CellParams(),
    "gabab": GABABParams(),
    "net": NetworkParams(),
    "stim": StimParams(),
    "stdp": STDParams(),
    "accept": AcceptanceParams(),
}
