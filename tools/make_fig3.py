"""WP-F: rebuild Figure 3.  No simulation.

(a) is re-plotted from a2_mode_map.csv (EO rows) with the same v2 mode criteria
    as Fig 4: amp = 0 is labelled intrinsic (there is no carrier, so no lock can
    be claimed), and the points are coloured by mode.
(b) keeps the exploratory extended scan, with the internal test id cropped off.

Output: figures/fig3_entrainment_v2.png
Usage:  python tools/make_fig3.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2 = os.path.join(ROOT, "stage2_network_gamma")
FIG = os.path.join(ROOT, "figures")
A2_CSV = os.path.join(S2, "a2_mode_map.csv")
EXT_PNG = os.path.join(FIG, "f_t25c_tacs_extended.png")

MODE_COLOR = {"intrinsic": "#9e9e9e", "1:2": "#ff7f0e", "3:2": "#9467bd",
              "1:1": "#2ca02c", "partial": "#f7c6c7", "silent": "#cccccc"}


def classify(amp, rate, f_peak, prom):
    if amp <= 0:
        return "intrinsic"
    if rate < 1.0:
        return "silent"
    if np.isnan(f_peak) or prom < 10:
        return "partial"
    if abs(f_peak - 40.0) <= 1.0 and abs(rate - 40.0) <= 5.0:
        return "1:1"
    if abs(f_peak - 20.0) <= 1.0 and abs(rate - 20.0) <= 5.0:
        return "1:2"
    if abs(f_peak - 60.0) <= 2.5:
        return "3:2"
    return "partial"


def main():
    with open(A2_CSV, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["state"] == "EO"]

    fig = plt.figure(figsize=(16, 5.6))
    fig.suptitle("Figure 3. 40 Hz tACS drives the network through three locked regimes",
                 fontsize=14, y=0.99, weight="bold")

    ax = fig.add_axes([0.055, 0.13, 0.36, 0.72])
    amps = sorted({float(r["amp_tacs"]) for r in rows})
    rng = np.random.default_rng(0)
    for amp in amps:
        sel = [r for r in rows if float(r["amp_tacs"]) == amp]
        xs = amp + rng.uniform(-0.12, 0.12, len(sel))
        ys = [float(r["f_peak"]) for r in sel]
        cols = [MODE_COLOR[classify(amp, float(r["rate"]), float(r["f_peak"]),
                                    float(r["prom"]))] for r in sel]
        ax.scatter(xs, ys, s=48, c=cols, edgecolor="white", linewidth=0.6, zorder=3)
        med = np.nanmedian(ys)
        ax.plot([amp - 0.22, amp + 0.22], [med, med], color="#333333",
                lw=2.0, zorder=2)
        mode = max({classify(amp, float(r["rate"]), float(r["f_peak"]),
                             float(r["prom"])) for r in sel},
                   key=lambda m: sum(1 for r in sel
                                     if classify(amp, float(r["rate"]),
                                                 float(r["f_peak"]),
                                                 float(r["prom"])) == m))
        ax.text(amp, 2.0, mode, ha="center", fontsize=8.5, color="#333333")

    ax.axhline(40, color="#2ca02c", ls="--", lw=1.0, alpha=0.8)
    ax.text(4.45, 40.6, "carrier 40 Hz", ha="right", fontsize=8, color="#2ca02c")
    ax.axhline(20, color="#ff7f0e", ls=":", lw=1.0, alpha=0.8)
    ax.set_xlabel("tACS amplitude (µA/cm²)")
    ax.set_ylabel("Peak frequency f_peak (Hz)")
    ax.set_title("(a) locked frequency vs amplitude, EO (5 doses × 3 seeds)", fontsize=11)
    ax.set_xticks(amps)
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(0, 72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_b = fig.add_axes([0.48, 0.13, 0.50, 0.72])
    if os.path.exists(EXT_PNG):
        img = Image.open(EXT_PNG)
        w, h = img.size
        ax_b.imshow(np.asarray(img.crop((0, int(0.055 * h), w, h))))
        ax_b.axis("off")
    ax_b.set_title("(b) exploratory extended amplitude scan (0.05–10 µA/cm²)",
                   loc="left", fontsize=11)

    out = os.path.join(FIG, "fig3_entrainment_v2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved: {out}\n")

    print("amp  n  f_peak median [min-max]  modes")
    for amp in amps:
        sel = [r for r in rows if float(r["amp_tacs"]) == amp]
        ys = np.array([float(r["f_peak"]) for r in sel])
        modes = [classify(amp, float(r["rate"]), float(r["f_peak"]), float(r["prom"]))
                 for r in sel]
        print(f"{amp:3g} {len(sel):3d}  {np.nanmedian(ys):6.2f} "
              f"[{np.nanmin(ys):.1f}-{np.nanmax(ys):.1f}]  {dict((m, modes.count(m)) for m in set(modes))}")


if __name__ == "__main__":
    main()
