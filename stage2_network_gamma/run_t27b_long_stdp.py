"""T2.7b: 长时程 STDP 维持（60s 刺激 + 120s 观察，无 TimedArray 内存爆炸）。

修复：
1. 放弃 TimedArray，用 network_operation 实时计算 tACS 电流和相位，避免内存溢出。
2. 刺激期拉长到 60s（模拟 iTBS 的一个 block），让 LTD 充分累积。
3. 观察期 120s，验证 P2 是否达到 +25%（1.25x）。
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, run, defaultclock, seed, ms, mV, mS, cm, uA,
                    Hz, SpikeMonitor, Network, Synapses, network_operation)
from common.networks import make_ing
from common.params import PARAMS

N = 100
I_MU = 0.85
SEED = 42
DT_MS = 0.025

T_WARMUP = 2000.0
T_BASE = 10000.0
T_STIM = 60000.0    # 60s 刺激
T_POST = 120000.0   # 120s 观察
T_TOTAL = T_WARMUP + T_BASE + T_STIM + T_POST

F_TACS = 40.0
AMP_TACS = 2.0
W_INIT = 1.0
W_MIN = 0.3
W_MAX = 2.0
A_LTP = -0.0003     # LTD
TAU_DECAY = 120000.0 # 观察期内极慢的恢复（模拟长期维持）

start_scope()
defaultclock.dt = DT_MS * ms
seed(SEED)

# 1. 构建网络
cell = PARAMS["cell"]
saved = (cell.g_Kv3, cell.g_Kdr)
cell.g_Kv3, cell.g_Kdr = 40.0, 20.0
try:
    G, _, gap = make_ing(N, i_mu=I_MU, seed=SEED)
finally:
    cell.g_Kv3, cell.g_Kdr = saved

# 2. STDP 突触
syn_ii = Synapses(G, G, model="""
    dg/dt = -g / (10*ms) : 1 (clock-driven)
    I_syn_post = w_syn * g_syn_each * g * (-75*mV - v_post) : amp/meter**2 (summed)
    w_syn : 1
    g_syn_each : siemens/meter**2 (constant)
""", on_pre="""g += 1
    g = clip(g, 0, 1)""")
syn_ii.connect(condition="i != j", p=0.2)
g_syn_each_val = (0.1 * mS / (cm**2)) / (N * 0.2)
syn_ii.g_syn_each = g_syn_each_val
syn_ii.w_syn = W_INIT

# 3. 给 G 注入 tACS 电流（用一个 dummy Synapses 保持接口统一，或者直接通过 state monitor 控制）
# 为了不爆内存，我们用 network_operation 在每一毫秒更新 G.I_inj
# 但 G 的 I_inj 是 constant，不能直接改，所以用一个外部电流源注入到 I_GABA_B
tacs_syn = Synapses(G, G, model="""
    I_GABA_B_post = I_stim : amp/meter**2 (summed)
    I_stim : amp/meter**2
""")
tacs_syn.connect(condition="i == j")
tacs_syn.I_stim = 0.0 * uA / (cm**2)

weight_hist = []
rate_hist = []

# 4. 实时更新机制（电流 + 权重 + 记录）
@network_operation(dt=1.0 * ms)
def dynamic_update(t):
    t_ms = float(t / ms)
    
    # 注入 tACS 电流
    if T_WARMUP + T_BASE <= t_ms < T_WARMUP + T_BASE + T_STIM:
        I_val = AMP_TACS * np.sin(2 * np.pi * F_TACS * t_ms / 1000.0)
        tacs_syn.I_stim = I_val * uA / (cm**2)
    else:
        tacs_syn.I_stim = 0.0 * uA / (cm**2)
        
    # STDP 更新
    w_now = syn_ii.w_syn[:]
    if T_WARMUP + T_BASE <= t_ms < T_WARMUP + T_BASE + T_STIM:
        phase = (t_ms / 25.0) % 1.0
        g_phi = max(0.0, np.cos(2 * np.pi * (phase - 0.25)))
        if g_phi > 0:
            dw = A_LTP * g_phi * (w_now - W_MIN) / (W_INIT - W_MIN)
            w_new = np.clip(w_now + dw, W_MIN, W_MAX)
            syn_ii.w_syn[:] = w_new
    elif t_ms >= T_WARMUP + T_BASE + T_STIM:
        # 撤除后缓慢恢复
        w_new = W_INIT + (w_now - W_INIT) * np.exp(-1.0 / TAU_DECAY)
        syn_ii.w_syn[:] = w_new
        
    # 记录
    if int(t_ms) % 5000 == 0:
        weight_hist.append((t_ms, float(np.mean(syn_ii.w_syn[:]))))

sp = SpikeMonitor(G)
net = Network(G, syn_ii, gap, tacs_syn, sp, dynamic_update)
net.run(T_TOTAL * ms)

# 5. 分析
all_t = np.array(sp.t / ms)

def calc_gamma_power(t_segment, window_ms=5000.0):
    if len(t_segment) < 10: return 0.0
    t0, t1 = t_segment.min(), min(t_segment.max(), t_segment.min() + window_ms)
    bins = np.arange(t0, t1 + 1, 1.0)
    hist, _ = np.histogram(t_segment, bins=bins)
    if len(hist) < 2: return 0.0
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, d=1e-3)
    band = (freqs >= 30) & (freqs <= 50)
    return spec[band].sum()

t_base = all_t[(all_t >= T_WARMUP + 5000) & (all_t < T_WARMUP + T_BASE)]
t_post_start = all_t[(all_t >= T_WARMUP + T_BASE + T_STIM) & (all_t < T_WARMUP + T_BASE + T_STIM + 10000)]
t_post_mid = all_t[(all_t >= T_WARMUP + T_BASE + T_STIM + 55000) & (all_t < T_WARMUP + T_BASE + T_STIM + 65000)]
t_post_end = all_t[(all_t >= T_TOTAL - 10000) & (all_t < T_TOTAL)]

p_base = calc_gamma_power(t_base)
p_post_start = calc_gamma_power(t_post_start)
p_post_mid = calc_gamma_power(t_post_mid)
p_post_end = calc_gamma_power(t_post_end)

print("=" * 60)
print("T2.7b 长时程 STDP 维持验证 (60s 刺激 + 120s 观察)")
print("=" * 60)
print(f"  γ 功率基线         = {p_base:.1f}")
print(f"  γ 功率撤除后 0-10s  = {p_post_start:.1f}  (vs 基线: {p_post_start/p_base:.2f}x)")
print(f"  γ 功率撤除后 55-65s = {p_post_mid:.1f}  (vs 基线: {p_post_mid/p_base:.2f}x)")
print(f"  γ 功率撤除后 110-120s= {p_post_end:.1f}  (vs 基线: {p_post_end/p_base:.2f}x)")
print()

if weight_hist:
    t_wh = np.array([w[0] for w in weight_hist])
    w_wh = np.array([w[1] for w in weight_hist])
    print(f"  权重初始: {W_INIT:.3f}, 最低: {w_wh.min():.3f}, 最终: {w_wh[-1]:.3f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1, ax2 = axes
ax1.plot(t_wh / 1000.0, w_wh, 'b-', lw=2)
ax1.axvspan((T_WARMUP + T_BASE)/1000.0, (T_WARMUP + T_BASE + T_STIM)/1000.0, color='red', alpha=0.1, label='tACS on')
ax1.axhline(W_INIT, color='gray', ls=':', label='w_init')
ax1.set_ylabel("Mean synaptic weight w")
ax1.set_title("T2.7b: long-term phase-gated STDP (60s stim, 120s post)")
ax1.legend()

window = 2000.0
bins = np.arange(0, T_TOTAL + window, window)
rates = np.zeros(len(bins) - 1)
for j in range(len(bins) - 1):
    rates[j] = np.sum((all_t >= bins[j]) & (all_t < bins[j+1])) / (N * window / 1000.0)
bin_centers = bins[:-1] + window / 2.0
ax2.plot(bin_centers / 1000.0, rates, 'k-', lw=1)
ax2.axvspan((T_WARMUP + T_BASE)/1000.0, (T_WARMUP + T_BASE + T_STIM)/1000.0, color='red', alpha=0.1)
ax2.set_ylabel("Population rate (Hz)")
ax2.set_xlabel("Time (s)")
plt.tight_layout()

fig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "f_t27b_long_stdp.png")
plt.savefig(fig_path, dpi=150)
print(f"\nsaved: {fig_path}")