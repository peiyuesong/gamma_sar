"""T2.6: EO/EC 状态依赖反转（终极防爆版 v2）。

修复：
1. Python 层面细分 Euler (dt=0.001ms) 预计算 GABA_B 电流，避免 Synapses 内 ODE 爆炸
2. tACS + GABA_B 合并为单一 TimedArray，只用一个 summed 变量
3. G_GABA_B_val 从 0.02 降到 0.005，避免刺激后完全死机
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, run, defaultclock, seed, ms, mV, mS, cm,
                    uA, mM, Hz, StateMonitor, SpikeMonitor, Network, Synapses,
                    TimedArray)
from common.networks import make_ing
from common.params import PARAMS

N = 100
I_MU = 0.75
SEED = 42
DT_MS = 0.025

T_BASE = 1000.0
T_STIM = 2000.0
T_POST = 1000.0
T_TOTAL = T_BASE + T_STIM + T_POST
N_STEPS = int(T_TOTAL / DT_MS) + 1

G_KV3_LIST = [40.0, 20.0]
STIM_MODES = ["Sham", "iTBS", "tACS+iTBS"]
AMP_TACS = 2.0
F_TACS = 40.0
G_GABA_B_val = 0.01  # 降低，避免完全死机

pg = PARAMS["gabab"]
n_coop = pg.n_coop
K1_g = pg.K1 * 1000.0
K2_g = pg.K2 * 1000.0
K3_eff_g = (pg.K3 / (pg.K_d ** (1.0 / n_coop))) * 1000.0
K4_g = pg.K4 * 1000.0

# ============ 预计算 GABA_B 电流轨迹 ============
def compute_gaba_b_traj():
    dt_fine = 0.001  # ms
    n_fine = int(T_TOTAL / dt_fine) + 1
    r_traj = np.zeros(n_fine)
    s_traj = np.zeros(n_fine)
    
    for i in range(1, n_fine):
        t_ms = i * dt_fine
        T_input = 0.0
        if T_BASE <= t_ms < T_BASE + T_STIM:
            t_rel = t_ms - T_BASE
            if (t_rel % 100.0) < 20.0:
                T_input = 1.0
        
        dr = (K1_g * T_input * (1 - r_traj[i-1]) - K2_g * r_traj[i-1]) * (dt_fine / 1000.0)
        ds = (K3_eff_g * r_traj[i-1] - K4_g * s_traj[i-1]) * (dt_fine / 1000.0)
        r_traj[i] = max(0.0, min(1.0, r_traj[i-1] + dr))
        s_traj[i] = max(0.0, s_traj[i-1] + ds)
    
    t_fine = np.arange(n_fine) * dt_fine
    t_coarse = np.arange(N_STEPS) * DT_MS
    s_coarse = np.interp(t_coarse, t_fine, s_traj)
    
    g_factor = s_coarse**n_coop / (s_coarse**n_coop + 1.0)
    return -G_GABA_B_val * g_factor * 25.0  # uA/cm2

I_GABA_B_TRAJ = compute_gaba_b_traj()

def run_sim(g_kv3, stim_mode):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(SEED)

    cell = PARAMS["cell"]
    saved = (cell.g_Kv3, cell.g_Kdr)
    cell.g_Kv3, cell.g_Kdr = g_kv3, 20.0
    try:
        G, syn_ii, gap = make_ing(N, i_mu=I_MU, seed=SEED)
    finally:
        cell.g_Kv3, cell.g_Kdr = saved

    objects = [G, syn_ii, gap]

    # 合并电流波形
    t_arr = np.arange(N_STEPS) * DT_MS
    I_wave = np.zeros(N_STEPS)

    if "tACS" in stim_mode:
        mask = (t_arr >= T_BASE) & (t_arr < T_BASE + T_STIM)
        I_wave[mask] += AMP_TACS * np.sin(2 * np.pi * F_TACS * t_arr[mask] / 1000.0)

    if "iTBS" in stim_mode:
        I_wave += I_GABA_B_TRAJ

    I_timed = TimedArray(I_wave * uA / (cm**2), dt=DT_MS * ms)

    stim_syn = Synapses(
        G, G,
        model="""
        I_GABA_B_post = I_stim : amp/meter**2 (summed)
        I_stim = I_timed(t) : amp/meter**2
        """,
    )
    stim_syn.connect(condition="i == j")
    stim_syn.namespace['I_timed'] = I_timed
    objects.append(stim_syn)

    sp = SpikeMonitor(G)
    objects.append(sp)
    net = Network(*objects)
    net.run(T_TOTAL * ms)

    all_t = np.array(sp.t / ms)

    def calc_gamma_power(t_segment):
        if len(t_segment) == 0:
            return 0.0
        t0, t1 = max(0, t_segment.min()), t_segment.max()
        if t1 <= t0:
            return 0.0
        bins = np.arange(t0, t1 + 1, 1.0)
        hist, _ = np.histogram(t_segment, bins=bins)
        if len(hist) < 2:
            return 0.0
        r = hist - hist.mean()
        nfft = max(8192, 2**int(np.ceil(np.log2(len(r)))))
        spec = np.abs(np.fft.rfft(r, nfft)) ** 2
        freqs = np.fft.rfftfreq(nfft, d=1e-3)
        band = (freqs >= 30) & (freqs <= 50)
        return spec[band].sum()

    t_stim = all_t[(all_t >= T_BASE + 500) & (all_t < T_BASE + T_STIM)] - (T_BASE + 500)
    t_post = all_t[(all_t >= T_BASE + T_STIM + 200) & (all_t < T_TOTAL)] - (T_BASE + T_STIM + 200)

    p_stim = calc_gamma_power(t_stim)
    p_post = calc_gamma_power(t_post)

    rate_stim = len(t_stim) / (N * (T_STIM - 500)/1000.0) if len(t_stim) > 0 else 0
    rate_post = len(t_post) / (N * (T_POST - 200)/1000.0) if len(t_post) > 0 else 0

    # 补充 f0 和 PLV
    if len(t_stim) > 0:
        bins = np.arange(0, T_STIM + 1, 1.0)
        hist, _ = np.histogram(t_stim, bins=bins)
        r = hist - hist.mean()
        spec = np.abs(np.fft.rfft(r, 8192)) ** 2
        freqs = np.fft.rfftfreq(8192, d=1e-3)
        band = (freqs >= 10) & (freqs <= 100)
        f0 = freqs[band][np.argmax(spec[band])] if spec[band].sum() > 0 else 0.0
        
        phase = (t_stim / 1000.0 * F_TACS) % 1.0
        plv = np.abs(np.mean(np.exp(1j * 2 * np.pi * phase)))
    else:
        f0, plv = 0.0, 0.0

    return p_stim, p_post, rate_stim, rate_post, f0, plv

print("=" * 80)
print("T2.6 EO/EC 状态依赖反转验证 (GABA_B=0.005)")
print("=" * 80)

results = {}
for g_kv3 in G_KV3_LIST:
    state_name = "EO (Kv3=40)" if g_kv3 == 40.0 else "EC (Kv3=20)"
    print(f"\n--- {state_name} ---")
    sham_p_stim = None
    for mode in STIM_MODES:
        p_stim, p_post, r_stim, r_post, f0, plv = run_sim(g_kv3, mode)
        results[(g_kv3, mode)] = (p_stim, p_post, r_stim, r_post, f0, plv)

        if mode == "Sham":
            sham_p_stim = p_stim

        ratio = p_stim / sham_p_stim if (sham_p_stim and sham_p_stim > 0) else 0
        print(f"  [{mode:9s}] γP={p_stim:10.1f} ({ratio:5.2f}x) rate={r_stim:5.1f}Hz "
              f"f0={f0:5.1f} PLV={plv:.2f} | post rate={r_post:5.1f}Hz")

# 汇总
print("\n" + "=" * 80)
print("核心对比")
print("=" * 80)
eo_sham = results[(40.0, "Sham")][0]
eo_combo = results[(40.0, "tACS+iTBS")][0]
ec_sham = results[(20.0, "Sham")][0]
ec_combo = results[(20.0, "tACS+iTBS")][0]

print(f"  EO Sham γP    = {eo_sham:10.1f}")
print(f"  EO tACS+iTBS  = {eo_combo:10.1f}  (vs Sham: {eo_combo/eo_sham:.2f}x)")
print(f"  EC Sham γP    = {ec_sham:10.1f}")
print(f"  EC tACS+iTBS  = {ec_combo:10.1f}  (vs Sham: {ec_combo/ec_sham:.2f}x)")

reversal = "PASS" if (eo_combo > eo_sham and ec_combo < ec_sham) else "FAIL"
print(f"\n  状态依赖反转: [{reversal}]")
print(f"  （EO 联合 > Sham, EC 联合 < Sham 才成立）")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
modes = ["Sham", "iTBS", "tACS+iTBS"]
x = np.arange(len(modes))
width = 0.35

for i, g_kv3 in enumerate(G_KV3_LIST):
    state_name = "EO (gKv3=40)" if g_kv3 == 40.0 else "EC (gKv3=20)"
    sham_p = results[(g_kv3, "Sham")][0]
    ratios = [results[(g_kv3, m)][0] / sham_p if sham_p > 0 else 0 for m in modes]
    ax.bar(x + (i - 0.5)*width, ratios, width, label=state_name)

ax.axhline(1.0, color='gray', ls='--')
ax.set_ylabel("Stimulus-period gamma power (vs Sham)")
ax.set_xticks(x)
ax.set_xticklabels(modes)
ax.set_title("T2.6: tACS+iTBS effect in EO vs EC states")
ax.legend()
plt.tight_layout()
fig_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures", "f_t26_eo_ec_reversal.png"
)
plt.savefig(fig_path, dpi=150)
print(f"\nsaved: {fig_path}")