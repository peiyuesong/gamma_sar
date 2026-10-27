"""A1 pilot: does 40 Hz tACS reopen a *true* gamma rhythm at I_mu = 0.85?

Why this script exists
----------------------
The v0.1 intervention figures (T2.6d) were run at I_mu = 0.80, where the
endogenous rhythm is ~18 Hz (1:1 mode, below the gamma band).  The declared
"sham gamma power" therefore only measured harmonic leakage.  At
I_mu = 0.85 the same network sits in the 2:1 ING mode at ~36 Hz, i.e. a
genuine gamma rhythm, so suppression/reopening can be tested against a
real gamma baseline.

This pilot only asks a yes/no question with 1 seed before committing to the
full 10-seed run:
  - EO  : g_Kv3 = 40 (gamma anchor, f0 ~38-40 Hz)
  - EC  : g_Kv3 = 0  (unified with the text; no more 0-vs-20 split)
  - conditions: Sham / iTBS / tACS+iTBS concurrent / tACS+iTBS sequential

Protocol layer only: network block is make_ing() verbatim, drive and
stimulus timing are the only things changed here.

Outputs
-------
  stage2_network_gamma/a1_pilot.csv
  stdout table: per-condition rate / f_peak / 30-50 Hz power / peak prominence
                in two windows (W1 = 1000-3000 ms, W2 = 3000-5000 ms)
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brian2
from brian2 import (start_scope, run, defaultclock, seed, ms, uA, cm,
                    SpikeMonitor, Network, Synapses, TimedArray)
from common.networks import make_ing
from common.params import PARAMS

N = 100
I_MU = 0.85          # A1: true-gamma drive
DT_MS = 0.025
T_BASE = 1000.0      # intervention onset
T_STIM = 2000.0      # iTBS / tACS train duration
T_TOTAL = 6000.0     # long enough for the sequential protocol
N_STEPS = int(T_TOTAL / DT_MS) + 1

G_KV3_LIST = [40.0, 0.0]     # EO, EC
AMP_TACS = 2.0
F_TACS = 40.0
# Frozen working dose.  The archived a1_pilot.csv was produced at g = 0.005, before
# the dose scan showed that 0.005 only slows the 2:1 rhythm (rate -18%) instead of
# silencing it; 0.02 saturates (98.5% firing-rate suppression).  Re-running this
# script now reproduces the frozen protocol, not the historical pilot.
G_GABAB = 0.02

CONDITIONS = ["Sham", "iTBS", "Concurrent", "Sequential"]

WINDOWS = {"W1": (1000.0, 3000.0), "W2": (3000.0, 5000.0)}

# Protocol metadata, registered next to the CSV so the Methods text can quote
# exactly what was simulated (requested by the text layer).
# 20 ms pulses every 100 ms = 10 Hz, 20 pulses across the 2 s train.  The working
# dose is g_GABA_B = 0.02 (the 0.005 used in v0.1 was retired by the dose scan).
ITBS_PROTOCOL = ("10 Hz (20 ms every 100 ms), 20 pulses over 2 s; "
                 "g_GABA_B = 0.02 mS/cm2 at the working point")
META = dict(
    script="run_a1_pilot.py",
    python=sys.version.split()[0],
    brian2=brian2.__version__,
    i_mu=I_MU, n_pv=N, dt_ms=DT_MS,
    g_gabab=G_GABAB, f_tacs=F_TACS, amp_tacs=AMP_TACS,
    itbs_protocol=ITBS_PROTOCOL,
    conditions=CONDITIONS,
    windows=WINDOWS,
    g_kv3={"EO": 40.0, "EC": 0.0},
    note=("Sequential = tACS 2000 ms after iTBS offset; with Destexhe kinetics "
          "(r decay ~1/K2 ~ 830 ms, s decay ~1/K4 ~ 130 ms) the network has "
          "essentially recovered by then, so Sequential is expected to approach "
          "Sham. Concurrent is the informative arm."),
)


def band_stats(t_spikes, t0, t1):
    """Rate + PSD diagnostics of the population rate in [t0, t1) ms."""
    seg = t_spikes[(t_spikes >= t0) & (t_spikes < t1)]
    dur_s = (t1 - t0) / 1000.0
    rate = len(seg) / (N * dur_s)
    out = {"rate": rate, "f_peak": float("nan"), "power": 0.0, "prom": 0.0}
    if len(seg) < 50:
        return out

    bins = np.arange(0.0, (t1 - t0) + 1.0, 1.0)   # 1 ms bins
    hist, _ = np.histogram(seg - t0, bins=bins)
    if len(hist) < 8:
        return out
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r, 8192)) ** 2
    freqs = np.fft.rfftfreq(8192, d=1e-3)          # 1 ms bins -> 1 kHz

    band = (freqs >= 30.0) & (freqs <= 50.0)
    search = (freqs >= 20.0) & (freqs <= 80.0)
    out["power"] = float(spec[band].sum())
    if search.sum() > 0:
        idx = np.argmax(spec[search])
        out["f_peak"] = float(freqs[search][idx])
        med = float(np.median(spec[search]))
        out["prom"] = float(spec[search][idx] / med) if med > 0 else 0.0
    return out


def run_sim(g_kv3, condition, seed_val, g_gabab=G_GABAB, amp_tacs=AMP_TACS,
            t_end=None, tacs_window=None):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(seed_val)

    cell = PARAMS["cell"]
    saved = (cell.g_Kv3, cell.g_Kdr)
    cell.g_Kv3, cell.g_Kdr = g_kv3, 20.0
    try:
        G, syn_ii, gap = make_ing(N, i_mu=I_MU, seed=seed_val)
    finally:
        cell.g_Kv3, cell.g_Kdr = saved

    objects = [G, syn_ii, gap]

    t_end = T_TOTAL if t_end is None else float(t_end)
    n_steps = int(t_end / DT_MS) + 1
    t_arr = np.arange(n_steps) * DT_MS
    I_wave = np.zeros(n_steps)

    # --- tACS: concurrent with iTBS, during the wash-out phase, or alone ---
    # tACS_only_* arms are the control the pilot lacked: same stimulus, no iTBS,
    # so "reopening" can be separated from plain entrainment of a normal network.
    # tacs_window=(start_ms, dur_ms) overrides the condition-derived timing, which
    # is what the wash-out experiments need (delay after iTBS offset).
    if tacs_window is not None:
        tacs_start, tacs_dur = tacs_window
        mask = (t_arr >= tacs_start) & (t_arr < tacs_start + tacs_dur)
    elif condition in ("Concurrent", "tACS_only_early"):
        mask = (t_arr >= T_BASE) & (t_arr < T_BASE + T_STIM)
    elif condition in ("Sequential", "tACS_only_late"):
        mask = (t_arr >= T_BASE + T_STIM) & (t_arr < T_BASE + 2 * T_STIM)
    else:
        mask = np.zeros_like(t_arr, dtype=bool)
    I_wave[mask] += amp_tacs * np.sin(2 * np.pi * F_TACS * t_arr[mask] / 1000.0)

    # --- iTBS surrogate: 20 ms GABA_B pulses every 100 ms for 2 s ---
    if condition in ("iTBS", "Concurrent", "Sequential"):
        dt_fine = 0.001
        # GABA_B is negligible by offset + 5 s (r ~ e^-6). Cap the fine-grained
        # trajectory there so long post-stimulus runs do not allocate tens of
        # millions of samples; behaviour is unchanged for t_end <= 8 s.
        t_traj_end = min(t_end, T_BASE + T_STIM + 5000.0)
        n_fine = int(t_traj_end / dt_fine) + 1
        r_traj = np.zeros(n_fine)
        s_traj = np.zeros(n_fine)
        t_fine = np.arange(n_fine) * dt_fine

        pg = PARAMS["gabab"]
        n_coop = pg.n_coop
        K1_g = pg.K1 * 1000.0
        K2_g = pg.K2 * 1000.0
        K3_eff_g = (pg.K3 / (pg.K_d ** (1.0 / n_coop))) * 1000.0
        K4_g = pg.K4 * 1000.0

        # NB: loop variable must not be named `i` -- brian2 resolves `i` in the
        # caller's namespace when building synaptic connect conditions, which
        # emits a spurious "resolution_conflict" warning after the loop.
        for k in range(1, n_fine):
            t_ms = k * dt_fine
            T_val = 0.0
            if T_BASE <= t_ms < T_BASE + T_STIM:
                if (t_ms - T_BASE) % 100.0 < 20.0:
                    T_val = 1.0
            dr = (K1_g * T_val * (1 - r_traj[k - 1]) - K2_g * r_traj[k - 1]) * (dt_fine / 1000.0)
            ds = (K3_eff_g * r_traj[k - 1] - K4_g * s_traj[k - 1]) * (dt_fine / 1000.0)
            r_traj[k] = max(0.0, min(1.0, r_traj[k - 1] + dr))
            s_traj[k] = max(0.0, min(1.0, s_traj[k - 1] + ds))

        n_coarse = int(t_traj_end / DT_MS) + 1
        s_coarse = np.interp(t_arr[:n_coarse], t_fine, s_traj)
        g_factor = s_coarse ** n_coop / (s_coarse ** n_coop + 1.0)
        I_wave[:n_coarse] += -g_gabab * g_factor * 25.0

    I_timed = TimedArray(I_wave * uA / (cm ** 2), dt=DT_MS * ms)
    stim_syn = Synapses(G, G, model="""
        I_GABA_B_post = I_stim : amp/meter**2 (summed)
        I_stim = I_timed(t) : amp/meter**2
    """)
    stim_syn.connect(condition="i == j")
    stim_syn.namespace['I_timed'] = I_timed
    objects.append(stim_syn)

    sp = SpikeMonitor(G)
    objects.append(sp)
    net = Network(*objects)
    net.run(t_end * ms)

    return np.array(sp.t / ms)


def main(seed_list):
    rows = []
    print("=" * 96)
    print(f"A1 pilot: I_mu={I_MU}, N={N}, dt={DT_MS} ms, seeds={seed_list}")
    print("=" * 96)
    for g_kv3 in G_KV3_LIST:
        state = "EO" if g_kv3 == 40.0 else "EC"
        for cond in CONDITIONS:
            for sd in seed_list:
                t_sp = run_sim(g_kv3, cond, sd)
                for wname, (t0, t1) in WINDOWS.items():
                    st = band_stats(t_sp, t0, t1)
                    rows.append(dict(state=state, g_kv3=g_kv3, condition=cond,
                                     seed=sd, window=wname, **st))
                    print(f"{state} | {cond:10s} | seed={sd:2d} | {wname} | "
                          f"rate={st['rate']:6.2f} Hz | f_peak={st['f_peak']:6.2f} Hz | "
                          f"P30_50={st['power']:12.1f} | prom={st['prom']:5.2f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "a1_pilot.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("state,g_kv3,condition,seed,window,rate_hz,f_peak_hz,power_30_50,prominence,"
                 "i_mu,dt_ms,g_gabab,f_tacs,amp_tacs,itbs_protocol\n")
        for r in rows:
            fh.write(f"{r['state']},{r['g_kv3']},{r['condition']},{r['seed']},{r['window']},"
                     f"{r['rate']:.4f},{r['f_peak']:.4f},{r['power']:.4f},{r['prom']:.4f},"
                     f"{I_MU},{DT_MS},{G_GABAB},{F_TACS},{AMP_TACS},\"{ITBS_PROTOCOL}\"\n")
    print(f"\nsaved CSV: {csv_path}")

    meta_path = os.path.join(out_dir, "a1_pilot_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({**META, "seeds": seed_list}, fh, ensure_ascii=False, indent=2)
    print(f"saved meta: {meta_path}")

    # --- compact comparison table: W2 (post-intervention) per state ---
    print("\nSummary (W2 = 3000-5000 ms, mean over seeds):")
    print(f"{'state':6s} {'condition':12s} {'rate':>8s} {'f_peak':>8s} {'P30_50':>12s} {'prom':>6s}")
    for g_kv3 in G_KV3_LIST:
        state = "EO" if g_kv3 == 40.0 else "EC"
        for cond in CONDITIONS:
            sel = [r for r in rows if r["state"] == state and r["condition"] == cond and r["window"] == "W2"]
            print(f"{state:6s} {cond:12s} {np.mean([r['rate'] for r in sel]):8.2f} "
                  f"{np.nanmean([r['f_peak'] for r in sel]):8.2f} "
                  f"{np.mean([r['power'] for r in sel]):12.1f} "
                  f"{np.mean([r['prom'] for r in sel]):6.2f}")


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [42]
    main(seeds)
