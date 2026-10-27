"""A1 phase 2: wash-out reopening, spontaneous recovery, threshold robustness.

Why (and a correction to the previous reading)
---------------------------------------------
The dose scan's W2 (3000-5000 ms) was read as "the network stays silent after
GABA_B has decayed, therefore it is bistable".  That inference is wrong: s is
driven by r and saturates (s_ss = K3_eff*r/K4 ~ 7.4*r, clipped at 1), so with
r's decay time ~830 ms the conductance stays at full strength (g_factor ~0.5,
i.e. -0.25 uA/cm2 at g=0.02) until r drops below ~0.135, i.e. ~1.4 s after
offset.  W2 stops at offset+2 s.  So the scan never observed the network after
GABA_B was actually gone.

Two consequences:
  * the "bistable silence" claim is untested -- it needs post >= offset+3 s;
  * a real wash-out reopening test must start tACS *after* the residual
    conductance is gone (offset + 2 s), which the v0.1 "Sequential" arm did not
    do (it started at offset + 0 s).

Arms (all EO, gKv3 = 40, I_mu = 0.85, dt = 0.025; iTBS = 1000-3000 ms)
--------------------------------------------------------------------------
  C1 tACS_only      g=0.00 amp=2.0   normal network + tACS (control; W2 reopened
                                     in v0.1 was never compared against this)
  C2 washout_amp2   g=0.02 amp=2.0   iTBS, then tACS at 5000-7000 (offset+2 s)
  C3 washout_amp3   g=0.02 amp=3.0   same, stronger carrier
  C4 conc_amp4      g=0.02 amp=4.0   fallback: can a stronger concurrent carrier
                                     pull 3:2 back to 1:1?
  C5 recovery       g=0.02, no tACS, 11 s of post-stimulus rate in 1 s bins
                                     -> is silence transient or metastable?
  C6/C7 itbs seed 1/2  g=0.02        is the silencing threshold seed-robust?

Outputs: a1_phase2.csv, a1_phase2_recovery.csv, a1_phase2_meta.json
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_a1_pilot import run_sim, band_stats, N, I_MU, DT_MS, ITBS_PROTOCOL

G_KV3 = 40.0
F_TACS = 40.0

WASH = {"iTBS": (1000.0, 3000.0), "washout": (3000.0, 5000.0),
        "tACS": (5000.0, 7000.0), "post": (7000.0, 9000.0)}

ARMS = [
    dict(label="tACS_only", condition="tACS_only_early", g=0.00, amp=2.0,
         seed=42, t_end=5000.0, tacs_window=(1000.0, 2000.0),
         windows={"pre": (0.0, 1000.0), "tACS": (1000.0, 3000.0),
                  "post": (3000.0, 5000.0)}),
    dict(label="washout_amp2", condition="Sequential", g=0.02, amp=2.0,
         seed=42, t_end=9000.0, tacs_window=(5000.0, 2000.0), windows=WASH),
    dict(label="washout_amp3", condition="Sequential", g=0.02, amp=3.0,
         seed=42, t_end=9000.0, tacs_window=(5000.0, 2000.0), windows=WASH),
    dict(label="conc_amp4", condition="Concurrent", g=0.02, amp=4.0,
         seed=42, t_end=6000.0, tacs_window=None,
         windows={"iTBS": (1000.0, 2200.0), "tACS": (2200.0, 3000.0),
                  "post": (3000.0, 6000.0)}),
    dict(label="itbs_seed1", condition="iTBS", g=0.02, amp=0.0,
         seed=1, t_end=6000.0, tacs_window=None,
         windows={"iTBS": (1000.0, 3000.0), "post": (3000.0, 6000.0)}),
    dict(label="itbs_seed2", condition="iTBS", g=0.02, amp=0.0,
         seed=2, t_end=6000.0, tacs_window=None,
         windows={"iTBS": (1000.0, 3000.0), "post": (3000.0, 6000.0)}),
]

RECOVERY = dict(label="recovery", condition="iTBS", g=0.02, amp=0.0,
                seed=42, t_end=14000.0, tacs_window=None)


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

    print("=" * 112)
    print(f"A1 phase 2 | I_mu={I_MU}, gKv3={G_KV3} (EO), dt={DT_MS} ms, iTBS 1000-3000 ms")
    print("=" * 112)

    for arm in ARMS:
        t_sp = run_sim(G_KV3, arm["condition"], arm["seed"], g_gabab=arm["g"],
                       amp_tacs=arm["amp"], t_end=arm["t_end"],
                       tacs_window=arm["tacs_window"])
        for wname, (t0, t1) in arm["windows"].items():
            st = band_stats(t_sp, t0, t1)
            verdict = classify(st["rate"], st["f_peak"], st["prom"])
            rows.append(dict(arm=arm["label"], condition=arm["condition"],
                             g_gabab=arm["g"], amp_tacs=arm["amp"], seed=arm["seed"],
                             window=wname, verdict=verdict, **st))
            print(f"{arm['label']:14s} | seed={arm['seed']:2d} | g={arm['g']:5.3f} "
                  f"amp={arm['amp']:3.1f} | {wname:8s} | rate={st['rate']:6.2f} Hz | "
                  f"f_peak={st['f_peak']:6.2f} Hz | prom={st['prom']:8.2f} | {verdict}")

    # --- spontaneous recovery: 1 s bins, 0-14 s ---
    print("\n" + "-" * 112)
    print("Spontaneous recovery after a silencing iTBS pulse (g=0.02, no tACS)")
    print("-" * 112)
    t_sp = run_sim(G_KV3, RECOVERY["condition"], RECOVERY["seed"],
                   g_gabab=RECOVERY["g"], amp_tacs=0.0, t_end=RECOVERY["t_end"])
    rec_rows = []
    baseline = len(t_sp[(t_sp >= 0) & (t_sp < 1000)]) / (N * 1.0)
    print(f"{'t (s)':>6s} {'rate (Hz)':>10s} {'vs baseline':>12s}")
    for s in range(int(RECOVERY["t_end"] / 1000.0)):
        seg = t_sp[(t_sp >= s * 1000.0) & (t_sp < (s + 1) * 1000.0)]
        rate = len(seg) / (N * 1.0)
        rec_rows.append(dict(t_s=s, rate=rate, baseline=baseline,
                             frac=rate / baseline if baseline > 0 else 0.0))
        print(f"{s:6d} {rate:10.2f} {rate / baseline:12.2f}")

    off_s = 3.0   # iTBS offset in seconds
    t50 = next((r["t_s"] for r in rec_rows
                if r["t_s"] >= off_s and r["frac"] >= 0.5), None)
    t90 = next((r["t_s"] for r in rec_rows
                if r["t_s"] >= off_s and r["frac"] >= 0.9), None)
    print(f"\nbaseline (0-1 s) = {baseline:.2f} Hz")
    print(f"t50 (first 1-s bin >= 50% baseline after offset) = {t50} s")
    print(f"t90 (first 1-s bin >= 90% baseline after offset) = {t90} s")

    out_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(out_dir, "a1_phase2.csv")
    keys = ["arm", "condition", "g_gabab", "amp_tacs", "seed", "window",
            "rate", "f_peak", "power", "prom", "verdict"]
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + ",i_mu,dt_ms,g_kv3\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + f",{I_MU},{DT_MS},{G_KV3}\n")
    print(f"\nsaved CSV: {csv_path}")

    rec_path = os.path.join(out_dir, "a1_phase2_recovery.csv")
    with open(rec_path, "w", encoding="utf-8") as fh:
        fh.write("t_s,rate_hz,baseline_hz,frac\n")
        for r in rec_rows:
            fh.write(f"{r['t_s']},{r['rate']:.4f},{r['baseline']:.4f},{r['frac']:.4f}\n")
    print(f"saved CSV: {rec_path}")

    meta_path = os.path.join(out_dir, "a1_phase2_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(dict(script="run_a1_phase2.py", i_mu=I_MU, g_kv3=G_KV3,
                       dt_ms=DT_MS, f_tacs=F_TACS, n_pv=N,
                       itbs_protocol=ITBS_PROTOCOL,
                       arms=[{k: v for k, v in a.items() if k != "windows"} for a in ARMS],
                       recovery=RECOVERY, t50_s=t50, t90_s=t90,
                       baseline_hz=baseline),
                  fh, ensure_ascii=False, indent=2)
    print(f"saved meta: {meta_path}")


if __name__ == "__main__":
    main()
