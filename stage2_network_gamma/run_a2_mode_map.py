"""A2: (g_GABA_B x tACS amplitude) locked-mode map for the frozen protocol.

What this figure is
-------------------
New Fig 4.  Not a "power" heatmap -- a *mode* map.  For each (dose, amplitude)
cell the network either falls silent, locks 1:1 to the 40 Hz carrier, locks 3:2
at 60 Hz, locks 1:2 at 20 Hz, or simply recovers its intrinsic 2:1 rhythm.  That
categorical outcome is robust in a way a fixed-band power integral is not (the
band integral changed by 16x with window length; the mode did not).

Protocol (frozen 2026-10-22)
---------------------------
  iTBS surrogate   20 ms every 100 ms for 2 s, 1000-3000 ms, g_GABA_B from the grid
  tACS             +2 s after iTBS offset, 5000-7000 ms, amplitude from the grid
  read-out         mean rate / f_peak / prominence over the tACS window
  run length       9000 ms

Grid: g in {0, 0.005, 0.01, 0.02, 0.05} x amp in {0, 1, 2, 3, 4}
      x state {EO gKv3=40, EC gKv3=0} x seeds {0,1,2}  = 150 runs (~75 min)

Design notes
------------
  * Every row is written as soon as it is computed, and re-running skips
    (state, g, amp, seed) rows already on disk -- a crash at run 140 keeps 139.
  * --save-spikes stores the raw spike train per cell (~60 KB), so the map can
    be re-derived at another window length without re-simulating.  The WP-C
    post-mortem showed why that matters.
  * The amp=0 column is the no-stimulation control: the network recovers its
    own 2:1 rhythm in that window, which is exactly what the "recovered" cells
    should look like.

Usage
  python stage2_network_gamma/run_a2_mode_map.py            # full grid, resumable
  python stage2_network_gamma/run_a2_mode_map.py --state EO # one state only
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_a1_pilot import run_sim, band_stats, N, I_MU, DT_MS, ITBS_PROTOCOL

STATES = {"EO": 40.0, "EC": 0.0}
G_GRID = [0.0, 0.005, 0.01, 0.02, 0.05]
AMP_GRID = [0.0, 1.0, 2.0, 3.0, 4.0]
SEEDS = [0, 1, 2]

F_TACS = 40.0
T_END = 9000.0
TACS_WINDOW = (5000.0, 2000.0)      # start, duration (iTBS offset + 2 s)
READOUT = (5000.0, 7000.0)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "a2_mode_map.csv")
SPK_DIR = os.path.join(OUT_DIR, "a2_spikes")

FIELDS = ["state", "g_kv3", "g_gabab", "amp_tacs", "seed",
          "rate", "f_peak", "prom", "mode", "wall_s"]
SAVE_SPIKES = False


def classify(rate, f_peak, prom):
    if rate < 1.0:
        return "silent"
    if np.isnan(f_peak) or prom < 3:
        return "no_peak"
    if abs(f_peak - 40.0) <= 1.5:
        return "1:1@40"
    if abs(f_peak - 60.0) <= 2.5:
        return "3:2@60"
    if abs(f_peak - 20.0) <= 1.5:
        return "1:2@20"
    if 34.0 <= f_peak <= 39.0:
        return "2:1~36"
    return "other"


def done_keys(path):
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        if not fh.readline():
            return keys
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= 5:
                try:
                    keys.add((p[0], float(p[2]), float(p[3]), int(p[4])))
                except ValueError:
                    continue
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="", help="EO, EC, or empty for both")
    ap.add_argument("--save-spikes", action="store_true")
    ap.add_argument("--g", default="", help="comma list override, e.g. 0.02")
    ap.add_argument("--amp", default="", help="comma list override, e.g. 2,4")
    args = ap.parse_args()

    global SAVE_SPIKES
    SAVE_SPIKES = args.save_spikes

    states = [args.state.upper()] if args.state.upper() in STATES else list(STATES)
    g_grid = [float(x) for x in args.g.split(",")] if args.g else G_GRID
    amp_grid = [float(x) for x in args.amp.split(",")] if args.amp else AMP_GRID

    if SAVE_SPIKES:
        os.makedirs(SPK_DIR, exist_ok=True)

    done = done_keys(CSV_PATH)
    new_file = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", encoding="utf-8", newline="")
    if new_file:
        fh.write(",".join(FIELDS) + "\n")
        fh.flush()

    total = len(states) * len(g_grid) * len(amp_grid) * len(SEEDS)
    print("=" * 104)
    print(f"A2 mode map | states={states} g={g_grid} amp={amp_grid} seeds={SEEDS}")
    print(f"total {total} runs; {len(done)} already on disk; dt={DT_MS} ms, I_mu={I_MU}")
    print("=" * 104)

    t_start = time.time()
    for state in states:
        g_kv3 = STATES[state]
        for g in g_grid:
            for amp in amp_grid:
                for sd in SEEDS:
                    if (state, g, amp, sd) in done:
                        continue
                    t0 = time.time()
                    t_sp = run_sim(g_kv3, "Sequential" if g > 0 else "tACS_only_late",
                                   sd, g_gabab=g, amp_tacs=amp,
                                   t_end=T_END, tacs_window=TACS_WINDOW)
                    wall = time.time() - t0
                    st = band_stats(t_sp, *READOUT)
                    mode = classify(st["rate"], st["f_peak"], st["prom"])
                    fh.write(f"{state},{g_kv3},{g},{amp},{sd},{st['rate']:.4f},"
                             f"{st['f_peak']:.4f},{st['prom']:.4f},{mode},{wall:.1f}\n")
                    fh.flush()
                    if SAVE_SPIKES:
                        np.savez_compressed(
                            os.path.join(SPK_DIR, f"{state}_g{g:g}_a{amp:g}_s{sd}.npz"),
                            t_ms=t_sp.astype(np.float32), state=np.array(state),
                            g_gabab=np.array(g), amp_tacs=np.array(amp),
                            seed=np.array(sd), dt_ms=np.array(DT_MS),
                            t_end=np.array(T_END), readout=np.array(READOUT))
                    print(f"{state} | g={g:<5g} amp={amp:<3g} | seed={sd} | "
                          f"rate={st['rate']:6.2f} f_peak={st['f_peak']:6.2f} "
                          f"prom={st['prom']:9.2f} | {mode:8s} | {wall:.0f}s")
    fh.close()

    # ---- mode table: majority across seeds, ties shown ----
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) == len(FIELDS):
                rows.append(p)
    print("\n" + "=" * 104)
    for state in states:
        print(f"\nmode map [{state}] (majority of {len(SEEDS)} seeds; x = seeds split)")
        print("g\\amp  " + "".join(f"{a:>10g}" for a in amp_grid))
        for g in g_grid:
            cells = []
            for amp in amp_grid:
                modes = [r[8] for r in rows
                         if r[0] == state and float(r[2]) == g and float(r[3]) == amp]
                if not modes:
                    cells.append("-")
                    continue
                uniq = {m: modes.count(m) for m in set(modes)}
                best = max(uniq, key=uniq.get)
                cells.append(best if len(uniq) == 1 else best + "*")
            print(f"{g:<6g} " + "".join(f"{c:>10s}" for c in cells))
    print(f"\nelapsed {(time.time() - t_start) / 60:.1f} min")

    meta_path = os.path.join(OUT_DIR, "a2_mode_map_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(dict(script="run_a2_mode_map.py", i_mu=I_MU, n_pv=N, dt_ms=DT_MS,
                       f_tacs=F_TACS, t_end_ms=T_END,
                       tacs_window=list(TACS_WINDOW), readout=list(READOUT),
                       states=STATES, g_grid=g_grid, amp_grid=amp_grid,
                       seeds=SEEDS, itbs_protocol=ITBS_PROTOCOL,
                       save_spikes=SAVE_SPIKES),
                  fh, ensure_ascii=False, indent=2)
    print(f"saved CSV: {CSV_PATH}")
    print(f"saved meta: {meta_path}")


if __name__ == "__main__":
    main()
