"""DEPRECATED -- superseded by run_c_stdp.py (which adds resume, --time, and the
correct effective dt).  Kept for history only; do not run this file.

STDP pilot: does the anti-phase window (LTP) produce a *real* post-stimulus
30-50 Hz peak, and does the original window (LTD) reproduce across seeds?

Context
-------
v0.1 reported a "+21% post-stimulus gamma elevation" from a single run (n=1),
and the trace in Fig 5 shows no visible 30-50 Hz oscillation at all.  Before
deciding whether to keep a P2-flavoured claim, two questions must be answered
with seeds:
  1. Is there a genuine spectral peak in 30-50 Hz after stimulation, or is the
     band power only noise/harmonic leakage?
  2. Does the effect reproduce across 10 seeds?

Arms
----
  ltd : G(phi) = max(0,  cos(2*pi*(phase-0.25))),  A = -0.0003  (original)
  ltp : G(phi) = max(0, -cos(2*pi*(phase-0.25))),  A = -0.0003  (anti-phase;
        the inverted window makes the same update push w *up*, i.e. LTP)

Note on direction (important): the synapse is inhibitory (GABA_A).  Weaker
inhibition -> disinhibition -> more firing/gamma; stronger inhibition -> less.
So the LTD arm, not the LTP arm, is the one that can raise gamma in this
network.  Both arms are run and reported; the decision rule is applied to the
data, not assumed.

This reuses the T2.7 plasticity implementation verbatim (w_syn multiplier,
network_operation at 1 ms, same update formula); only the sign/sign-window
changes, plus 10 seeds and a post-stimulus PSD panel.

Outputs
-------
  stage2_network_gamma/stdp_pilot.csv
  stdout per-seed table + summary
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brian2
from brian2 import (start_scope, ms, mV, mS, cm, uA, SpikeMonitor, Network,
                    Synapses, network_operation, defaultclock, seed, clip)
from common.networks import make_ing
from common.params import PARAMS

N = 100
I_MU = 0.85
G_KV3 = 40.0
DT_MS = 0.05          # pilot: half the cost of the 0.025 ms T2.7b run

T_WARMUP = 2000.0
T_BASE = 5000.0
T_STIM = 30000.0
T_POST = 60000.0
T_TOTAL = T_WARMUP + T_BASE + T_STIM + T_POST

F_TACS = 40.0
AMP_TACS = 2.0
W_INIT = 1.0
W_MIN = 0.3
W_MAX = 2.0
A_MAG = 0.0003
TAU_DECAY = 120000.0

POST_WIN = (25000.0, 45000.0)     # 25-45 s after stim offset

# Plasticity metadata, registered with the CSV so the Methods text can quote
# the implementation as it actually runs (not as v0.1 described it).
META = dict(
    script="run_stdp_pilot.py",
    python=sys.version.split()[0],
    brian2=brian2.__version__,
    i_mu=I_MU, g_kv3=G_KV3, n_pv=N, dt_ms=DT_MS,
    f_tacs=F_TACS, amp_tacs=AMP_TACS,
    w_init=W_INIT, w_min=W_MIN, w_max=W_MAX, a_mag=A_MAG, tau_decay_ms=TAU_DECAY,
    stim_s=T_STIM / 1000.0, post_window_s=[POST_WIN[0] / 1000.0, POST_WIN[1] / 1000.0],
    rule=("uniform update every 1 ms from the tACS phase, "
          "dw = sign * A * max(0, +-cos(2*pi*(phase-0.25))) * (w-w_min)/(w_init-w_min), "
          "clipped to [w_min, w_max]; NOT pre/post spike-timing dependent"),
    arms={"ltd": "max(0, +cos)", "ltp": "max(0, -cos)"},
)


def psd_stats(t_spikes, t0, t1):
    seg = t_spikes[(t_spikes >= t0) & (t_spikes < t1)]
    dur_s = (t1 - t0) / 1000.0
    rate = len(seg) / (N * dur_s)
    out = {"rate": rate, "power": 0.0, "f_peak": float("nan"), "prom": 0.0}
    if len(seg) < 100:
        return out
    bins = np.arange(0.0, (t1 - t0) + 1.0, 1.0)
    hist, _ = np.histogram(seg - t0, bins=bins)
    if len(hist) < 16:
        return out
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r, 16384)) ** 2
    freqs = np.fft.rfftfreq(16384, d=1e-3)
    band = (freqs >= 30.0) & (freqs <= 50.0)
    search = (freqs >= 20.0) & (freqs <= 80.0)
    out["power"] = float(spec[band].sum())
    idx = np.argmax(spec[search])
    out["f_peak"] = float(freqs[search][idx])
    med = float(np.median(spec[search]))
    out["prom"] = float(spec[search][idx] / med) if med > 0 else 0.0
    return out


def run_one(arm, seed_val):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(seed_val)

    cell = PARAMS["cell"]
    saved = (cell.g_Kv3, cell.g_Kdr)
    cell.g_Kv3, cell.g_Kdr = G_KV3, 20.0
    try:
        G, _, gap = make_ing(N, i_mu=I_MU, seed=seed_val)
    finally:
        cell.g_Kv3, cell.g_Kdr = saved

    # --- T2.7 synapse variant (weighted GABA_A), unchanged from T2.7b ---
    syn_ii = Synapses(G, G, model="""
        dg/dt = -g / (10*ms) : 1 (clock-driven)
        I_syn_post = w_syn * g_syn_each * g * (-75*mV - v_post) : amp/meter**2 (summed)
        w_syn : 1
        g_syn_each : siemens/meter**2 (constant)
    """, on_pre="""g += 1
        g = clip(g, 0, 1)""")
    syn_ii.connect(condition="i != j", p=0.2)
    syn_ii.g_syn_each = (0.1 * mS / (cm ** 2)) / (N * 0.2)
    syn_ii.w_syn = W_INIT

    tacs_syn = Synapses(G, G, model="""
        I_GABA_B_post = I_stim : amp/meter**2 (summed)
        I_stim : amp/meter**2
    """)
    tacs_syn.connect(condition="i == j")
    tacs_syn.I_stim = 0.0 * uA / (cm ** 2)

    weight_hist = []

    sign = -1.0 if arm == "ltd" else 1.0   # dw sign relative to the window

    @network_operation(dt=1.0 * ms)
    def dynamic_update(t):
        t_ms = float(t / ms)
        stim_on = (T_WARMUP + T_BASE) <= t_ms < (T_WARMUP + T_BASE + T_STIM)

        if stim_on:
            tacs_syn.I_stim = (AMP_TACS * np.sin(2 * np.pi * F_TACS * t_ms / 1000.0)
                               * uA / (cm ** 2))
        else:
            tacs_syn.I_stim = 0.0 * uA / (cm ** 2)

        w_now = syn_ii.w_syn[:]
        if stim_on:
            phase = (t_ms / 25.0) % 1.0
            if arm == "ltd":
                g_phi = max(0.0, np.cos(2 * np.pi * (phase - 0.25)))
            else:
                g_phi = max(0.0, -np.cos(2 * np.pi * (phase - 0.25)))
            if g_phi > 0:
                dw = sign * A_MAG * g_phi * (w_now - W_MIN) / (W_INIT - W_MIN)
                syn_ii.w_syn[:] = np.clip(w_now + dw, W_MIN, W_MAX)
        elif t_ms >= (T_WARMUP + T_BASE + T_STIM):
            w_new = W_INIT + (w_now - W_INIT) * np.exp(-1.0 / TAU_DECAY)
            syn_ii.w_syn[:] = w_new

        if int(t_ms) % 2000 == 0:
            weight_hist.append((t_ms, float(np.mean(syn_ii.w_syn[:]))))

    sp = SpikeMonitor(G)
    net = Network(G, syn_ii, gap, tacs_syn, sp, dynamic_update)
    net.run(T_TOTAL * ms)

    all_t = np.array(sp.t / ms)
    stim_off = T_WARMUP + T_BASE + T_STIM
    base = psd_stats(all_t, T_WARMUP, T_WARMUP + T_BASE)
    # Two post windows: the late one is the pre-registered P2 window, the early
    # one guards against the effect peaking before 25 s (weight recovers with
    # tau ~120 s, so an early transient would be missed by the late window).
    early = psd_stats(all_t, stim_off, stim_off + 20000.0)
    late = psd_stats(all_t, stim_off + POST_WIN[0], stim_off + POST_WIN[1])

    w_hist = np.array([w for _, w in weight_hist]) if weight_hist else np.array([np.nan])
    return dict(arm=arm, seed=seed_val,
                w_nadir=float(np.nanmin(w_hist)), w_final=float(w_hist[-1]),
                base_rate=base["rate"], post_rate=late["rate"],
                early_rate=early["rate"],
                base_power=base["power"], post_power=late["power"],
                early_power=early["power"],
                base_peak=base["f_peak"], post_peak=late["f_peak"],
                early_peak=early["f_peak"],
                post_prom=late["prom"], early_prom=early["prom"])


def main(seed_list):
    rows = []
    print("=" * 104)
    print(f"STDP pilot: I_mu={I_MU}, gKv3={G_KV3}, dt={DT_MS} ms, "
          f"stim={T_STIM/1000:.0f}s, post window={POST_WIN[0]/1000:.0f}-{POST_WIN[1]/1000:.0f}s")
    print("=" * 104)
    for arm in ("ltd", "ltp"):
        for sd in seed_list:
            r = run_one(arm, sd)
            rows.append(r)
            print(f"{arm} | seed={sd:2d} | w: {r['w_nadir']:.3f}->{r['w_final']:.3f} | "
                  f"rate {r['base_rate']:5.1f}->{r['post_rate']:5.1f} Hz | "
                  f"f_peak {r['post_peak']:5.1f} Hz | prom={r['post_prom']:5.2f} | "
                  f"P30_50 early x{r['early_power']/r['base_power']:.2f} "
                  f"late x{r['post_power']/r['base_power']:.2f}"
                  if r["base_power"] > 0 else f"{arm} | seed={sd:2d} | degenerate")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "stdp_pilot.csv")
    keys = ["arm", "seed", "w_nadir", "w_final", "base_rate", "post_rate",
            "base_power", "post_power", "base_peak", "post_peak", "post_prom",
            "early_rate", "early_power", "early_peak", "early_prom"]
    extra = dict(i_mu=I_MU, dt_ms=DT_MS, w_min=W_MIN, w_max=W_MAX,
                 a_mag=A_MAG, tau_decay_ms=TAU_DECAY)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys + list(extra)) + "\n")
        for r in rows:
            fh.write(",".join([str(r[k]) for k in keys] +
                              [str(v) for v in extra.values()]) + "\n")
    print(f"\nsaved CSV: {csv_path}")

    meta_path = os.path.join(out_dir, "stdp_pilot_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({**META, "seeds": seed_list}, fh, ensure_ascii=False, indent=2)
    print(f"saved meta: {meta_path}")

    print("\nSummary (mean +/- sd over seeds):")
    for arm in ("ltd", "ltp"):
        sel = [r for r in rows if r["arm"] == arm]
        late_r = [r["post_power"] / r["base_power"] for r in sel if r["base_power"] > 0]
        early_r = [r["early_power"] / r["base_power"] for r in sel if r["base_power"] > 0]
        print(f"{arm}: P30_50 early {np.mean(early_r):.2f} +/- {np.std(early_r):.2f} | "
              f"late {np.mean(late_r):.2f} +/- {np.std(late_r):.2f} | "
              f"post f_peak = {np.nanmean([r['post_peak'] for r in sel]):.1f} Hz | "
              f"prom = {np.mean([r['post_prom'] for r in sel]):.2f} | "
              f"rate {np.mean([r['base_rate'] for r in sel]):.1f} -> "
              f"{np.mean([r['post_rate'] for r in sel]):.1f} Hz")


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]]
    if not seeds:
        seeds = [42]          # smoke test; pass 10 seeds for the real pilot
    main(seeds)
