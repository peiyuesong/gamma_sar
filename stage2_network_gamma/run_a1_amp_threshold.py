"""Where does the locked mode flip from 3:2 to 1:1 under GABA_B inhibition?

Phase 2 found that at g_GABA_B = 0.02 (silencing dose) a 2.0 uA/cm2 carrier locks
the network at 60.06 Hz (3:2) while 4.0 uA/cm2 locks it at 40.04 Hz (1:1).  That
is an Arnold-tongue boundary, and it matters for the protocol: the "ON switch"
frequency is not fixed by the carrier alone, it depends on how much inhibition
the carrier has to overcome.

This script locates that boundary on the steady-state window (2200-3000 ms, after
the GABA_B conductance has plateaued) and checks whether it is seed-robust.

Arms: g_GABA_B = 0.02, concurrent iTBS+tACS, EO, I_mu = 0.85, dt = 0.025
  amp in {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} at seed 42
  amp = 2.0 at seeds 1 and 2 (is the 3:2 branch reproducible?)

Outputs: a1_amp_threshold.csv, a1_amp_threshold_meta.json
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_a1_pilot import run_sim, band_stats, N, I_MU, DT_MS, ITBS_PROTOCOL

G_KV3 = 40.0
G_GABAB = 0.02
F_TACS = 40.0
T_END = 6000.0
TACS_WINDOW = (1000.0, 2000.0)      # concurrent with the iTBS pulse train
STEADY = (2200.0, 3000.0)           # GABA_B plateau, tACS on
POST = (3000.0, 6000.0)

SEED = 42
AMPS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
EXTRA_SEEDS = [(2.0, 1), (2.0, 2)]


def classify(rate, f_peak, prom):
    if rate < 1.0:
        return "silent"
    if np.isnan(f_peak) or prom < 3:
        return "no peak"
    if abs(f_peak - 40.0) <= 1.0:
        return "1:1 @ 40 Hz"
    if abs(f_peak - 60.0) <= 2.0:
        return "3:2 @ 60 Hz"
    if abs(f_peak - 20.0) <= 1.5:
        return "1:2 @ 20 Hz"
    return f"other @ {f_peak:.1f}"


def main():
    rows = []
    print("=" * 104)
    print(f"Locked-mode vs carrier amplitude | g_GABA_B={G_GABAB}, EO, I_mu={I_MU}")
    print(f"steady window = {STEADY[0]:.0f}-{STEADY[1]:.0f} ms (tACS on, GABA_B plateaued)")
    print("=" * 104)

    for amp, seed in [(a, SEED) for a in AMPS] + EXTRA_SEEDS:
        t_sp = run_sim(G_KV3, "Concurrent", seed, g_gabab=G_GABAB, amp_tacs=amp,
                       t_end=T_END, tacs_window=TACS_WINDOW)
        st = band_stats(t_sp, *STEADY)
        po = band_stats(t_sp, *POST)
        v = classify(st["rate"], st["f_peak"], st["prom"])
        rows.append(dict(amp_tacs=amp, seed=seed, window="steady", verdict=v,
                         **st))
        rows.append(dict(amp_tacs=amp, seed=seed, window="post", verdict="",
                         **po))
        print(f"amp={amp:4.1f} seed={seed:2d} | steady: rate={st['rate']:6.2f} "
              f"f_peak={st['f_peak']:6.2f} prom={st['prom']:8.2f} | {v:14s} "
              f"|| post: rate={po['rate']:6.2f} f_peak={po['f_peak']:6.2f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "a1_amp_threshold.csv")
    keys = ["amp_tacs", "seed", "window", "rate", "f_peak", "power", "prom", "verdict"]
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + ",g_gabab,i_mu,dt_ms,g_kv3\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) +
                     f",{G_GABAB},{I_MU},{DT_MS},{G_KV3}\n")
    print(f"\nsaved CSV: {csv_path}")

    meta_path = os.path.join(out_dir, "a1_amp_threshold_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(dict(script="run_a1_amp_threshold.py", g_gabab=G_GABAB,
                       i_mu=I_MU, g_kv3=G_KV3, dt_ms=DT_MS, f_tacs=F_TACS,
                       amps=AMPS, extra_seeds=EXTRA_SEEDS, t_end_ms=T_END,
                       tacs_window=list(TACS_WINDOW), steady_window=list(STEADY),
                       n_pv=N, itbs_protocol=ITBS_PROTOCOL),
                  fh, ensure_ascii=False, indent=2)
    print(f"saved meta: {meta_path}")


if __name__ == "__main__":
    main()
