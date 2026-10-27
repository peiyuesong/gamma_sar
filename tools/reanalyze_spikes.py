"""Re-analyse saved spike trains under several window lengths -- no simulation.

Why
---
The full WP-C run reported P30_50 ratios of x18-22 (LTD) and x8-12 (LTP) over a
20 s post window.  Both arms rose, which cannot be a plasticity-direction effect:
the two arms share the 30 s tACS train, so the common factor is either an
entrainment after-effect or a window-length artefact.  The v0.1 t27b run, which
used a 5 s window, reported only x1.21.  Both questions are answered by
recomputing the same saved spikes at 1 / 2 / 5 / 20 s.

Inputs
------
  stage2_network_gamma/c_spikes/{arm}_seed{n}.npz  (written by --save-spikes)
  stage2_network_gamma/c_stdp.csv                  (arm, seed, window constants)

Outputs
-------
  stage2_network_gamma/c_reanalysis.csv            tidy: arm, seed, wlen, window, ...
  stdout: per-arm medians of early/base and late/base for each window length,
          plus a 1 s-bin coherence profile after stimulation.

Usage:  python tools/reanalyze_spikes.py
"""
import glob
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPK_DIR = os.path.join(ROOT, "stage2_network_gamma", "c_spikes")
OUT_CSV = os.path.join(ROOT, "stage2_network_gamma", "c_reanalysis.csv")

N = 100
# 20 s is deliberately absent: T_BASE is 5 s, so a 20 s baseline would start at
# -13 s, silently zero-pad 65% of the histogram, deflate the baseline power and
# inflate every ratio by ~16x.  Windows must fit inside the pre-stimulus period.
WLENS = [1.0, 2.0, 5.0]            # seconds
LATE_OFFSET = 25.0                 # seconds after stim offset
BAND = (30.0, 50.0)
SEARCH = (20.0, 80.0)


def psd(t_sp, t0, t1):
    """Rate, band power, peak frequency and prominence in [t0, t1] ms."""
    seg = t_sp[(t_sp >= t0) & (t_sp < t1)]
    dur_s = (t1 - t0) / 1000.0
    rate = len(seg) / (N * dur_s)
    out = dict(rate=rate, power=0.0, f_peak=float("nan"), prom=0.0)
    if len(seg) < 20:
        return out
    bins = np.arange(0.0, (t1 - t0) + 1.0, 1.0)
    hist, _ = np.histogram(seg - t0, bins=bins)
    if len(hist) < 8:
        return out
    r = hist - hist.mean()
    spec = np.abs(np.fft.rfft(r)) ** 2
    freqs = np.fft.rfftfreq(len(r), d=1e-3)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    search = (freqs >= SEARCH[0]) & (freqs <= SEARCH[1])
    out["power"] = float(spec[band].sum())
    idx = np.argmax(spec[search])
    out["f_peak"] = float(freqs[search][idx])
    med = float(np.median(spec[search]))
    out["prom"] = float(spec[search][idx] / med) if med > 0 else 0.0
    return out


def main():
    files = sorted(glob.glob(os.path.join(SPK_DIR, "*.npz")))
    if not files:
        print(f"no spike files under {SPK_DIR}")
        return 1

    rows = []
    loaded = []          # (arm, seed, t_ms, stim_on, stim_off) -- reused below
    for path in files:
        d = np.load(path)
        t = d["t_ms"].astype(np.float64)
        arm = str(d["arm"])
        seed = int(d["seed"])
        stim_on = float(d["t_warmup"] + d["t_base"])
        stim_off = stim_on + float(d["t_stim"])
        loaded.append((arm, seed, t, stim_on, stim_off))

        for wl in WLENS:
            w = wl * 1000.0
            if stim_on - w < 0:
                print(f"  skip {arm} seed {seed} wlen {wl}s: baseline would start "
                      f"at {stim_on - w:.0f} ms")
                continue
            base = psd(t, stim_on - w, stim_on)
            early = psd(t, stim_off, stim_off + w)
            late = psd(t, stim_off + LATE_OFFSET * 1000.0,
                       stim_off + LATE_OFFSET * 1000.0 + w)
            for wname, st in (("base", base), ("early", early), ("late", late)):
                rows.append(dict(arm=arm, seed=seed, wlen_s=wl, window=wname,
                                 rate=st["rate"], power=st["power"],
                                 f_peak=st["f_peak"], prom=st["prom"]))

    # ---- tidy CSV ----
    keys = ["arm", "seed", "wlen_s", "window", "rate", "power", "f_peak", "prom"]
    with open(OUT_CSV, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"saved: {OUT_CSV}\n")

    # ---- summary: median ratio per arm x window length ----
    print(f"{'arm':4s} {'wlen':>5s} {'n':>3s} {'early/base':>22s} {'late/base':>22s} "
          f"{'base prom':>10s} {'post f_peak':>12s}")
    arms = sorted({r["arm"] for r in rows})
    for arm in arms:
        for wl in WLENS:
            sel = [r for r in rows if r["arm"] == arm and r["wlen_s"] == wl]
            base = {r["seed"]: r for r in sel if r["window"] == "base"}
            early = [r for r in sel if r["window"] == "early" and r["seed"] in base]
            late = [r for r in sel if r["window"] == "late" and r["seed"] in base]
            if not early:
                continue
            e = np.array([r["power"] / base[r["seed"]]["power"] for r in early
                          if base[r["seed"]]["power"] > 0])
            l = np.array([r["power"] / base[r["seed"]]["power"] for r in late
                          if base[r["seed"]]["power"] > 0])
            bp = np.median([base[r["seed"]]["prom"] for r in early])
            fp = np.median([r["f_peak"] for r in early])
            print(f"{arm:4s} {wl:5.0f} {len(e):3d} "
                  f"{np.median(e):9.2f} [{np.percentile(e,25):.2f}-{np.percentile(e,75):.2f}] "
                  f"{np.median(l):9.2f} [{np.percentile(l,25):.2f}-{np.percentile(l,75):.2f}] "
                  f"{bp:10.0f} {fp:12.1f}")

    # ---- 1 s-bin profile after offset: is the coherence transient? ----
    print("\nPost-offset coherence profile (1 s bins, median over seeds)")
    print(f"{'t after off':>12s} " + " ".join(f"{a:>18s}" for a in arms))
    for s in range(0, 20):
        cells = []
        for arm in arms:
            vals = [psd(t, off + s * 1000.0, off + (s + 1) * 1000.0)["prom"]
                    for a, _sd, t, _on, off in loaded if a == arm]
            cells.append(np.median(vals) if vals else float("nan"))
        print(f"{s:12d} " + " ".join(f"{c:18.0f}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
