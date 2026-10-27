"""T2.7a: 相位门控 STDP 概念验证（短时窗 17.5s）。

修复：
1. 使用 network_operation 进行加性 LTP 更新，避免 on_pre 字符串解析报错。
2. dt_update = 1.0 ms，彻底解决 25ms 采样与 40Hz 导致的混叠问题。
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, run, defaultclock, seed, ms, mV, mS, cm, uA,
                    Hz, SpikeMonitor, Network, Synapses, TimedArray, network_operation)
from common.networks import make_ing
from common.params import PARAMS

N, I_MU, SEED, DT_MS = 100, 0.85, 42, 0.025
T_WARMUP, T_BASE, T_STIM, T_POST = 500.0, 2000.0, 5000.0, 10000.0
T_TOTAL = T_WARMUP + T_BASE + T_STIM + T_POST

F_TACS, AMP_TACS, PHI_PEAK = 40.0, 2.0, 90.0
W_INIT, W_MIN, W_MAX = 1.0, 0.5, 2.0

A_LTP = 0.0003     # 每毫秒最大增量（乘了 1ms 的 dt）
TAU_DECAY = 10000.0  # 撤除后衰减时间常数 ms

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

# 2. tACS 注入
t_arr = np.arange(0, T_TOTAL, DT_MS)
I_wave = np.zeros(len(t_arr))
mask = (t_arr >= T_WARMUP + T_BASE) & (t_arr < T_WARMUP + T_BASE + T_STIM)
I_wave[mask] = AMP_TACS * np.sin(2 * np.pi * F_TACS * t_arr[mask] / 1000.0)
I_timed = TimedArray(I_wave * uA / (cm**2), dt=DT_MS * ms)

tacs_syn = Synapses(G, G, model="""
    I_GABA_B_post = I_stim : amp/meter**2 (summed)
    I_stim = I_timed(t) : amp/meter**2
""")
tacs_syn.connect(condition="i == j")
tacs_syn.namespace['I_timed'] = I_timed

# 3. STDP 突触（只保留变量和基础前馈）
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

# 4. 相位门控（1ms 采样精度，半波整流）
dt_update = 1.0
n_cycles = int(T_TOTAL / dt_update) + 1
t_cycle = np.arange(n_cycles) * dt_update
phi = (2 * np.pi * F_TACS * t_cycle / 1000.0) % (2 * np.pi)

# 只在 phi_peak 附近为正
G_phi = np.maximum(0.0, np.cos(phi - np.deg2rad(PHI_PEAK)))
G_phi_arr = np.where(
    (t_cycle >= T_WARMUP + T_BASE) & (t_cycle < T_WARMUP + T_BASE + T_STIM),
    G_phi, 0.0
)
G_timed = TimedArray(G_phi_arr, dt=dt_update * ms)

weight_hist = []

@network_operation(dt=dt_update * ms)
def update_weights(t):
    t_ms = float(t / ms)
    w_now = syn_ii.w_syn[:]
    
    if T_WARMUP + T_BASE <= t_ms < T_WARMUP + T_BASE + T_STIM:
        g_phi = float(G_timed(t))
        if g_phi > 0:
            # 加性 LTP，带饱和
            dw = A_LTP * g_phi * (W_MAX - w_now) / (W_MAX - W_INIT)
            w_new = np.clip(w_now + dw, W_MIN, W_MAX)
            syn_ii.w_syn[:] = w_new
            
    elif t_ms >= T_WARMUP + T_BASE + T_STIM:
        # 撤除后衰减
        w_new = W_INIT + (w_now - W_INIT) * np.exp(-dt_update / TAU_DECAY)
        syn_ii.w_syn[:] = w_new
        
    if int(t_ms) % 500 == 0:
        weight_hist.append((t_ms, float(np.mean(syn_ii.w_syn[:]))))

sp = SpikeMonitor(G)
net = Network(G, syn_ii, gap, tacs_syn, sp, update_weights)
net.run(T_TOTAL * ms)

# 5. 分析
all_t = np.array(sp.t / ms)

def calc_gamma_power(t_segment, window_ms=2000.0):
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

t_base = all_t[(all_t >= T_WARMUP) & (all_t < T_WARMUP + T_BASE)]
t_stim_end = all_t[(all_t >= T_WARMUP + T_BASE + T_STIM - 2000) & (all_t < T_WARMUP + T_BASE + T_STIM)]
t_post_start = all_t[(all_t >= T_WARMUP + T_BASE + T_STIM) & (all_t < T_WARMUP + T_BASE + T_STIM + 2000)]
t_post_end = all_t[(all_t >= T_TOTAL - 2000) & (all_t < T_TOTAL)]

p_base = calc_gamma_power(t_base)
p_stim_end = calc_gamma_power(t_stim_end)
p_post_start = calc_gamma_power(t_post_start)
p_post_end = calc_gamma_power(t_post_end)

print("=" * 60)
direction = "LTP" if A_LTP > 0 else "LTD"
print(f"T2.7a 相位门控 STDP 概念验证 (v4, {direction})")
print("=" * 60)
print(f"  γ 功率基线         = {p_base:.1f}")
print(f"  γ 功率刺激末期      = {p_stim_end:.1f}  (vs 基线: {p_stim_end/p_base:.2f}x)")
print(f"  γ 功率撤除后 0-2s   = {p_post_start:.1f}  (vs 基线: {p_post_start/p_base:.2f}x)")
print(f"  γ 功率撤除后 8-10s  = {p_post_end:.1f}  (vs 基线: {p_post_end/p_base:.2f}x)")
print()

if weight_hist:
    t_wh = np.array([w[0] for w in weight_hist])
    w_wh = np.array([w[1] for w in weight_hist])
    print(f"  权重初始: {W_INIT:.3f}, 峰值: {w_wh.max():.3f}, 最终: {w_wh[-1]:.3f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1, ax2 = axes
ax1.plot(t_wh, w_wh, 'b-', lw=2)
ax1.axvspan(T_WARMUP + T_BASE, T_WARMUP + T_BASE + T_STIM, color='red', alpha=0.1, label='tACS on')
ax1.axhline(W_INIT, color='gray', ls=':', label='w_init')
ax1.set_ylabel("Mean synaptic weight w")
ax1.set_title("T2.7a: phase-gated STDP concept (v4, LTD)")
ax1.legend()

window = 100.0
bins = np.arange(0, T_TOTAL + window, window)
rates = np.zeros(len(bins) - 1)
for j in range(len(bins) - 1):
    rates[j] = np.sum((all_t >= bins[j]) & (all_t < bins[j+1])) / (N * window / 1000.0)
bin_centers = bins[:-1] + window / 2.0
ax2.plot(bin_centers, rates, 'k-', lw=1)
ax2.axvspan(T_WARMUP + T_BASE, T_WARMUP + T_BASE + T_STIM, color='red', alpha=0.1)
ax2.set_ylabel("Population rate (Hz)")
ax2.set_xlabel("Time (ms)")

plt.tight_layout()
direction = "LTP" if A_LTP > 0 else "LTD"
fig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "figures", f"f_t27a_stdp_{direction}.png")
plt.savefig(fig_path, dpi=150)
print(f"\nsaved: {fig_path}")