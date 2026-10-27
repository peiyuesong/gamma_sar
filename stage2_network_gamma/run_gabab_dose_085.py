"""GABA_B dose-response at the true-gamma operating point (I_mu = 0.85, EO).

Why: the A1 pilot showed that at I_mu = 0.85 the iTBS surrogate used in v0.1
(g_GABA_B = 0.005 mS/cm2, 20 ms pulses at 5 Hz for 2 s) only slows the ING
rhythm (36.5 -> 31.7 Hz, rate -18%) instead of silencing it.  v0.1's 90%
suppression was measured at I_mu = 0.80, where the network sits near the
2:1/1:1 boundary and is far easier to shut down.

So the OFF switch either (a) needs a larger GABA_B conductance at 0.85, or
(b) is intrinsically state-dependent.  This script finds which, in 6 runs.

Arms (all EO, g_Kv3 = 40, seed 42):
  Sham                      no GABA_B, no tACS
  iTBS g=0.005/0.01/0.02/0.05   pulse train only
  Concurrent g=0.02         pulse train + 40 Hz tACS at 2 uA/cm2

Outputs: stage2_network_gamma/gabab_dose_085.csv + _meta.json
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_a1_pilot import (run_sim, band_stats, N, I_MU, DT_MS, G_KV3_LIST,
                          WINDOWS, ITBS_PROTOCOL)

G_KV3 = 40.0          # EO
SEED = 42
DOSE_LIST = [0.005, 0.01, 0.02, 0.05]
COMBO_G = 0.02
AMP_TACS, F_TACS = 2.0, 40.0


def main():
    rows = []

    arms = [("Sham", "Sham", 0.0)]
    arms += [(f"iTBS_g{g:g}", "iTBS", g) for g in DOSE_LIST]
    arms += [(f"Concurrent_g{COMBO_G:g}", "Concurrent", COMBO_G)]

    print("=" * 100)
    print(f"GABA_B dose-response at I_mu={I_MU}, gKv3={G_KV3} (EO), seed={SEED}")
    print("=" * 100)

    for label, condition, g in arms:
        t_sp = run_sim(G_KV3, condition, SEED, g_gabab=g)
        for wname, (t0, t1) in WINDOWS.items():
            st = band_stats(t_sp, t0, t1)
            rows.append(dict(arm=label, condition=condition, g_gabab=g,
                             window=wname, **st))
            print(f"{label:20s} | {wname} | rate={st['rate']:6.2f} Hz | "
                  f"f_peak={st['f_peak']:6.2f} Hz | prom={st['prom']:8.2f} | "
                  f"P30_50={st['power']:.3e}")

    # --- suppression relative to Sham, same window ---
    sham = {r["window"]: r["rate"] for r in rows if r["arm"] == "Sham"}
    print("\nSuppression vs Sham (by window):")
    for r in rows:
        if r["arm"] == "Sham":
            continue
        base = sham[r["window"]]
        supp = (1.0 - r["rate"] / base) * 100 if base > 0 else float("nan")
        r["suppression_pct"] = supp
        print(f"{r['arm']:20s} | {r['window']} | {supp:6.1f}%  "
              f"(rate {base:.1f} -> {r['rate']:.1f} Hz)")
    for r in rows:
        r.setdefault("suppression_pct", 0.0)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "gabab_dose_085.csv")
    keys = ["arm", "condition", "g_gabab", "window", "rate", "f_peak", "power", "prom",
            "suppression_pct"]
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + ",i_mu,dt_ms,g_kv3,seed\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) +
                     f",{I_MU},{DT_MS},{G_KV3},{SEED}\n")
    print(f"\nsaved CSV: {csv_path}")

    meta_path = os.path.join(out_dir, "gabab_dose_085_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(dict(script="run_gabab_dose_085.py", i_mu=I_MU, g_kv3=G_KV3,
                       seed=SEED, dt_ms=DT_MS, doses=DOSE_LIST, combo_g=COMBO_G,
                       amp_tacs=AMP_TACS, f_tacs=F_TACS, n_pv=N,
                       itbs_protocol=ITBS_PROTOCOL, windows=WINDOWS),
                  fh, ensure_ascii=False, indent=2)
    print(f"saved meta: {meta_path}")


if __name__ == "__main__":
    main()
