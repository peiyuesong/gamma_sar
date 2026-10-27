"""WP-B: frozen-protocol statistics, 4 conditions x 2 states x 10 seeds = 80 runs.

Protocol frozen on 2026-10-22 (see docs/program_2026-10.md, decision log):
  OFF : pulsed iTBS surrogate, 20 ms every 100 ms for 2 s, g_GABA_B = 0.02
  ON  : 40 Hz tACS at 2.0 uA/cm2, starting +2 s after iTBS offset (5000-7000 ms)
  run : 9 s total; windows iTBS 1000-3000 / washout 3000-5000 / tACS 5000-7000
        / post 7000-9000

Conditions
  Sham        neither stimulus          (baseline reference)
  iTBS        OFF only                  (also feeds the rebuilt Fig 2c)
  iTBS+tACS   OFF then ON at +2 s       (the frozen two-step protocol)
  tACS_only   40 Hz tACS 5000-7000      (control: same ON stimulus, no OFF)

Why tACS_only must stay: without it, the reopening arm cannot be separated from
plain entrainment of a network that was about to recover on its own.

Design notes
  * CSV is written incrementally and the script resumes: re-running skips
    (state, condition, seed) rows already on disk, so a crash at run 70 does not
    throw away the first 69.
  * Metrics: rate + f_peak + peak prominence.  The 30-50 Hz band integral is
    deliberately NOT reported -- f0 moves between modes and a fixed-band integral
    produces the "suppression raises gamma power" paradox.

Usage
  python stage2_network_gamma/run_b_stats.py           # both states
  python stage2_network_gamma/run_b_stats.py EO        # one state (resumable)
"""
import json
import os
import sys
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_a1_pilot import run_sim, band_stats, N, I_MU, DT_MS, ITBS_PROTOCOL

F_TACS = 40.0
G_GABAB = 0.02
AMP_TACS = 2.0
T_END = 9000.0
ITBS_ON, ITBS_OFF = 1000.0, 3000.0
TACS_START, TACS_DUR = 5000.0, 2000.0

WINDOWS = {
    "iTBS": (1000.0, 3000.0),
    "washout": (3000.0, 5000.0),
    "tACS": (5000.0, 7000.0),
    "post": (7000.0, 9000.0),
}

STATES = {"EO": 40.0, "EC": 0.0}
SEEDS = list(range(10))
CONDITIONS = ["Sham", "iTBS", "iTBS+tACS", "tACS_only"]

# how each condition maps onto run_sim()
#   condition name -> (sim condition string, g_gabab, tacs_window)
PROTOCOL = {
    "Sham":      ("Sham",       0.0,     None),
    "iTBS":      ("iTBS",       G_GABAB, None),
    "iTBS+tACS": ("Sequential", G_GABAB, (TACS_START, TACS_DUR)),
    "tACS_only": ("tACS_only_late", 0.0, (TACS_START, TACS_DUR)),
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "b_stats.csv")
FIELDS = ["state", "condition", "seed", "window", "rate", "f_peak", "prom",
          "g_gabab", "amp_tacs", "tacs_start"]


def done_keys(path):
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            return keys
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 3:
                try:
                    keys.add((parts[0], parts[1], int(parts[2])))
                except ValueError:
                    continue
    return keys


def main(states):
    keys = done_keys(CSV_PATH)
    new_file = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", encoding="utf-8", newline="")
    if new_file:
        fh.write(",".join(FIELDS) + "\n")
    fh.flush()

    print("=" * 108)
    print(f"WP-B frozen protocol | g={G_GABAB}, amp={AMP_TACS}, tACS {TACS_START:.0f}-"
          f"{TACS_START + TACS_DUR:.0f} ms, states={states}, seeds={SEEDS}")
    if keys:
        print(f"resuming: {len(keys)} runs already on disk")
    print("=" * 108)

    for state in states:
        g_kv3 = STATES[state]
        for cond in CONDITIONS:
            sim_cond, g, tacs_win = PROTOCOL[cond]
            for sd in SEEDS:
                if (state, cond, sd) in keys:
                    continue
                t_sp = run_sim(g_kv3, sim_cond, sd, g_gabab=g, amp_tacs=AMP_TACS,
                               t_end=T_END, tacs_window=tacs_win)
                for wname, (t0, t1) in WINDOWS.items():
                    st = band_stats(t_sp, t0, t1)
                    row = (f"{state},{cond},{sd},{wname},{st['rate']:.4f},"
                           f"{st['f_peak']:.4f},{st['prom']:.4f},"
                           f"{g},{AMP_TACS},{TACS_START:.0f}")
                    fh.write(row + "\n")
                fh.flush()
                print(f"{state} | {cond:10s} | seed={sd:2d} | "
                      f"iTBS {band_stats(t_sp, *WINDOWS['iTBS'])['rate']:5.2f} Hz | "
                      f"wash {band_stats(t_sp, *WINDOWS['washout'])['rate']:5.2f} Hz | "
                      f"tACS {band_stats(t_sp, *WINDOWS['tACS'])['rate']:5.2f} Hz | "
                      f"post {band_stats(t_sp, *WINDOWS['post'])['rate']:5.2f} Hz")
    fh.close()

    # ---- summary from whatever is on disk now ----
    data = {}
    with open(CSV_PATH, "r", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) < 7:
                continue
            data.setdefault((p[0], p[1], p[3]), []).append(
                dict(rate=float(p[4]), f_peak=float(p[5]), prom=float(p[6])))

    print("\n" + "=" * 108)
    print("Summary (mean +/- sd over seeds; f_peak is nan-aware)")
    print(f"{'state':5s} {'condition':11s} {'window':8s} {'rate':>16s} "
          f"{'f_peak':>14s} {'prom':>14s}")
    for state in states:
        for cond in CONDITIONS:
            for wname in WINDOWS:
                rows = data.get((state, cond, wname), [])
                if not rows:
                    continue
                r = np.array([x["rate"] for x in rows])
                f = np.array([x["f_peak"] for x in rows])
                p = np.array([x["prom"] for x in rows])
                print(f"{state:5s} {cond:11s} {wname:8s} "
                      f"{r.mean():7.2f}+/-{r.std():5.2f} "
                      f"{np.nanmean(f):6.2f}+/-{np.nanstd(f):4.2f} "
                      f"{p.mean():6.2f}+/-{p.std():5.2f}")

    print("\nWilcoxon (paired by seed, rate):")
    for state in states:
        def rates(cond, wname):
            return np.array([x["rate"] for x in data.get((state, cond, wname), [])])
        for wname in ("iTBS", "tACS"):
            a = rates("iTBS+tACS", wname)
            b = rates("Sham", wname)
            if len(a) == len(b) and len(a) >= 5 and not np.all(a == b):
                try:
                    _, p = wilcoxon(a, b)
                    print(f"  {state} | {wname:8s} | iTBS+tACS vs Sham: p={p:.4f}")
                except ValueError:
                    pass
        a = rates("tACS_only", "tACS")
        b = rates("iTBS+tACS", "tACS")
        if len(a) == len(b) and len(a) >= 5:
            try:
                _, p = wilcoxon(a, b)
                print(f"  {state} | tACS      | tACS_only vs iTBS+tACS: p={p:.4f}")
            except ValueError:
                pass

    meta_path = os.path.join(OUT_DIR, "b_stats_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(dict(script="run_b_stats.py", g_gabab=G_GABAB, amp_tacs=AMP_TACS,
                       i_mu=I_MU, dt_ms=DT_MS, n_pv=N, f_tacs=F_TACS,
                       itbs_protocol=ITBS_PROTOCOL, t_end_ms=T_END,
                       tacs_window=[TACS_START, TACS_DUR], windows=WINDOWS,
                       states=STATES, seeds=SEEDS, conditions=CONDITIONS),
                  fh, ensure_ascii=False, indent=2)
    print(f"\nsaved CSV: {CSV_PATH}")
    print(f"saved meta: {meta_path}")


if __name__ == "__main__":
    arg = sys.argv[1].upper() if len(sys.argv) > 1 else ""
    chosen = [arg] if arg in STATES else list(STATES)
    main(chosen)
