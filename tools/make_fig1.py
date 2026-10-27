"""WP-F: rebuild Figure 1 from frozen CSVs only.  No simulation.

The v0.1 figure carried internal workflow language baked into the image ("MS1",
"hygiene-corrected", "T2.1d"), which cannot be edited out by cropping, so both
panels are re-plotted from the raw scans:
  (a) t13_fI_surface.csv    -- f-I surface + steady-state V_end per g_Kv3
  (b) t21d_fine.csv         -- operating map: f0 by branch, P(2:1) with Wilson CI

Output: figures/fig1_validation_v2.png
Usage:  python tools/make_fig1.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
FI_CSV = os.path.join(ROOT, "stage1_single_cell", "t13_fI_surface.csv")
MAP_CSV = os.path.join(ROOT, "stage2_network_gamma", "t21d_fine.csv")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    fi = load(FI_CSV)
    mp = load(MAP_CSV)

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22,
                          left=0.06, right=0.985, top=0.90, bottom=0.08)
    ax_rate = fig.add_subplot(gs[0, 0])
    ax_vend = fig.add_subplot(gs[0, 1])
    ax_f0 = fig.add_subplot(gs[1, 0])
    ax_p = fig.add_subplot(gs[1, 1])

    fig.suptitle("Figure 1. Single-cell validation and the ING operating regime.",
                 fontsize=14, y=0.975, weight="bold")
    fig.text(0.06, 0.925, "(a) Traub–Miles PV neuron: f-I surface and steady state",
             fontsize=11.5, weight="bold")
    fig.text(0.06, 0.475, "(b) Network operating map: 2:1 vs 1:1 mode",
             fontsize=11.5, weight="bold")

    # ---- (a) f-I and V_end ----
    gvals = sorted({float(r["g_kv3_mScm2"]) for r in fi})
    colors = cm.viridis(np.linspace(0.0, 0.92, len(gvals)))
    for g, col in zip(gvals, colors):
        rows = sorted([r for r in fi if float(r["g_kv3_mScm2"]) == g],
                      key=lambda r: float(r["i_uA_cm2"]))
        i = np.array([float(r["i_uA_cm2"]) for r in rows])
        rate = np.array([float(r["rate_hz"]) for r in rows])
        vend = np.array([float(r["v_end_mV"]) for r in rows])
        blocked = np.array([int(r["blocked"]) for r in rows])
        ax_rate.plot(i, rate, "-o", ms=3, lw=1.4, color=col, label=f"{g:g}")
        ax_vend.plot(i, vend, "-o", ms=3, lw=1.4, color=col)
        if blocked.any():
            ax_vend.plot(i[blocked == 1], vend[blocked == 1], "x", ms=8,
                         mew=1.6, color="#d62728")

    ax_rate.set_xlabel("Injected current (µA/cm²)")
    ax_rate.set_ylabel("Firing rate (Hz)")
    # no per-axes title here: the row header above the panel already names it
    ax_rate.legend(title="g_Kv3 (mS/cm²)", fontsize=8, title_fontsize=8, ncol=2)
    ax_vend.set_xlabel("Injected current (µA/cm²)")
    ax_vend.set_ylabel("Steady-state V_end (mV)")
    ax_vend.axhline(-35, color="#d62728", ls=":", lw=1)
    for ax in (ax_rate, ax_vend):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ---- (b) operating map ----
    gsel = [0.0, 40.0]
    style = {0.0: (("#1f77b4", "#2ca02c"), "g_Kv3 = 0 (EC)"),
             40.0: (("#d62728", "#9467bd"), "g_Kv3 = 40 (EO)")}
    for g in gsel:
        # NB: t21d_fine.csv also holds a small Kdr scan (g_Kv3=40, Kdr in
        # {10,15,25}, only at I=1.00, 20 seeds each).  Without the Kdr filter
        # those 60 rows merge into the (g=40, I=1.00) cell -> n=80.
        rows = [r for r in mp if float(r["g_kv3"]) == g
                and float(r["g_kdr"]) == 20.0]
        ivals = sorted({float(r["i_mu"]) for r in rows})
        f21, f11, p21, lo, hi = [], [], [], [], []
        for i in ivals:
            sel = [r for r in rows if float(r["i_mu"]) == i]
            n = len(sel)
            k = sum(1 for r in sel if "2:1" in r["class"])
            p21.append(k / n)
            l, h = wilson(k, n)
            lo.append(l)
            hi.append(h)
            two = [float(r["f0_hz"]) for r in sel if "2:1" in r["class"]]
            one = [float(r["f0_hz"]) for r in sel if "1:1" in r["class"]]
            f21.append(np.median(two) if two else np.nan)
            f11.append(np.median(one) if one else np.nan)
        (c21, c11), label = style[g]
        ax_f0.plot(ivals, f21, "-o", ms=4, lw=1.6, color=c21,
                   label=f"{label}, 2:1")
        ax_f0.plot(ivals, f11, "--s", ms=4, lw=1.2, color=c11, alpha=0.85,
                   label=f"{label}, 1:1")
        ax_p.plot(ivals, p21, "-o", ms=4, lw=1.6, color=c21, label=label)
        ax_p.fill_between(ivals, lo, hi, color=c21, alpha=0.18, lw=0)

    ax_f0.axhspan(38, 42, color="#2ca02c", alpha=0.10, lw=0)
    ax_f0.text(max(ivals) * 0.995, 40.6, "tACS 40 Hz", ha="right",
               fontsize=8, color="#2ca02c")
    ax_f0.set_xlabel("Drive I_µ (µA/cm²)")
    ax_f0.set_ylabel("Fundamental frequency f0 (Hz)")
    ax_f0.set_title("Mode frequency by branch (median over seeds)", fontsize=11)
    ax_f0.legend(fontsize=7.5, ncol=2)

    ax_p.axhline(0.5, color="#555555", ls="--", lw=0.9)
    ax_p.text(0.695, 0.52, "majority (0.5)", fontsize=7.5, color="#555555",
              ha="left", va="bottom")
    ax_p.set_xlabel("Drive I_µ (µA/cm²)")
    ax_p.set_ylabel("P(2:1)")
    ax_p.set_ylim(-0.03, 1.03)
    ax_p.set_title("P(2:1) with Wilson 95% CI (20 seeds)", fontsize=11)
    ax_p.legend(fontsize=8)
    for ax in (ax_f0, ax_p):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = os.path.join(FIG, "fig1_validation_v2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved: {out}\n")

    for g in gsel:
        # NB: t21d_fine.csv also holds a small Kdr scan (g_Kv3=40, Kdr in
        # {10,15,25}, only at I=1.00, 20 seeds each).  Without the Kdr filter
        # those 60 rows merge into the (g=40, I=1.00) cell -> n=80.
        rows = [r for r in mp if float(r["g_kv3"]) == g
                and float(r["g_kdr"]) == 20.0]
        ivals = sorted({float(r["i_mu"]) for r in rows})
        print(f"g_Kv3={g:g}")
        for i in ivals:
            sel = [r for r in rows if float(r["i_mu"]) == i]
            k = sum(1 for r in sel if "2:1" in r["class"])
            two = [float(r["f0_hz"]) for r in sel if "2:1" in r["class"]]
            print(f"  I={i:<5.2f} n={len(sel):2d} P(2:1)={k/len(sel):4.2f} "
                  f"f0_2:1={np.median(two) if two else float('nan'):5.1f} Hz")


if __name__ == "__main__":
    main()
