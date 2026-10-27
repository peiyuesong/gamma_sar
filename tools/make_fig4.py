"""WP-F: rebuild Figure 4 from frozen CSVs only.  No simulation.

v0.1's Fig 4 plotted statistics from the pre-fix script at I=0.80, g=0.005,
EC gKv3=20, with "3.45x / 3.57x" hard-coded.  This rebuild uses b_stats.csv
(I=0.85, g=0.02, EC gKv3=0, n=10) and a2_mode_map.csv, and derives every number
on the figure from the data.

Two corrections over the first rebuild
--------------------------------------
1. Fold changes are computed against the *same state's* Sham, not EO's.  An
   earlier draft labelled the EC bars with EO's denominator (40.00/16.72=2.39
   instead of 40.00/19.03=2.10) -- the same cross-group mistake as the retired
   cross-window comparison.
2. Locked modes are re-derived from the raw rate/f_peak, not read from the CSV's
   `mode` column.  The old classifier put EC's *intrinsic* 38.6 Hz rhythm into
   the "1:1@40" bucket (|38.6-40| < 1.5), so the amp=0 column of a map with no
   carrier at all read "locked".  A 1:1 claim now requires the carrier to exist,
   |f_peak-40| <= 1 AND |rate-40| <= 5 (one spike per carrier cycle).

Panels
  (a) left  firing rate during iTBS (1-3 s): Sham / iTBS / iTBS+tACS
      right firing rate during tACS (5-7 s): Sham / iTBS+tACS / tACS only
  (b) locked-mode map, g_GABA_B x amplitude, EO and EC; '*' = seeds disagree.

Output: figures/fig4_switch_v2.png
Usage:  python tools/make_fig4.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2 = os.path.join(ROOT, "stage2_network_gamma")
FIG = os.path.join(ROOT, "figures")
B_CSV = os.path.join(S2, "b_stats.csv")
A2_CSV = os.path.join(S2, "a2_mode_map.csv")

STATES = ["EO", "EC"]
BLUE, ORANGE, GREEN, RED = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
CARRIER_HZ = 40.0

# mode -> (index, colour, text, white_text?)
MODES = [
    ("intrinsic", "#9e9e9e", "2:1†", True),
    ("1:2",       "#ff7f0e", "1:2",  True),
    ("3:2",       "#9467bd", "3:2",  True),
    ("1:1",       "#2ca02c", "1:1",  True),
    ("partial",   "#f7c6c7", "~",    False),
    ("silent",    "#e0e0e0", "–",    False),
    ("other",     "#ffffff", "?",    False),
]
MODE_INDEX = {name: i for i, (name, *_rest) in enumerate(MODES)}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def classify(amp, rate, f_peak, prom):
    """Carrier-aware mode.  Never claims a lock when no carrier was applied."""
    if amp <= 0:
        return "intrinsic"          # no carrier: this is the intrinsic rhythm
    if rate < 1.0:
        return "silent"
    if np.isnan(f_peak) or prom < 10:
        return "partial"        # activity without a clear spectral peak
    if abs(f_peak - CARRIER_HZ) <= 1.0 and abs(rate - CARRIER_HZ) <= 5.0:
        return "1:1"
    if abs(f_peak - 20.0) <= 1.0 and abs(rate - 20.0) <= 5.0:
        return "1:2"
    if abs(f_peak - 60.0) <= 2.5:
        return "3:2"
    return "partial"


def rate_of(rows, state, cond, window):
    return np.array([float(r["rate"]) for r in rows
                     if r["state"] == state and r["condition"] == cond
                     and r["window"] == window])


def peak_of(rows, state, cond, window):
    return np.array([float(r["f_peak"]) for r in rows
                     if r["state"] == state and r["condition"] == cond
                     and r["window"] == window])


def main():
    b = load(B_CSV)
    a2 = load(A2_CSV)

    fig = plt.figure(figsize=(15, 9.6))
    fig.suptitle("Figure 4. A two-step switch: iTBS silences the network, "
                 "40 Hz tACS re-entrains it.", fontsize=14, y=0.985, weight="bold")

    # ---------------- (a) ---------------- #
    groups = [
        ("(a) during iTBS (1–3 s)", "iTBS",
         [("Sham", "Sham", BLUE), ("iTBS", "iTBS", RED), ("iTBS+tACS", "iTBS+tACS", ORANGE)]),
        ("(b) during tACS (5–7 s)", "tACS",
         [("Sham", "Sham", BLUE), ("iTBS+tACS", "iTBS+tACS", ORANGE), ("tACS_only", "tACS only", GREEN)]),
    ]
    axes_a = [fig.add_axes([0.055, 0.56, 0.38, 0.34]),
              fig.add_axes([0.545, 0.56, 0.38, 0.34])]
    width = 0.36

    for ax, (title, window, conds) in zip(axes_a, groups):
        x = np.arange(len(conds))
        sham = {s: rate_of(b, s, "Sham", window) for s in STATES}
        denom = {s: (sham[s].mean() if len(sham[s]) else np.nan) for s in STATES}
        for k, state in enumerate(STATES):
            vals = [rate_of(b, state, c, window) for c, _l, _c in conds]
            means = [v.mean() for v in vals]
            sds = [v.std() for v in vals]
            ax.bar(x + (k - 0.5) * width, means, width, yerr=sds, capsize=3,
                   color=[c for _c, _l, c in conds], alpha=0.95 if k == 0 else 0.55,
                   edgecolor="white", linewidth=0.6,
                   hatch="" if k == 0 else "//")
            for xi, m in zip(x, means):
                # fold change always against the SAME state's Sham
                txt = f"{m:.2f}"
                d = denom[state]
                if xi > 0 and np.isfinite(d) and d > 0:
                    txt += f"\n(×{m / d:.2f})"
                ax.text(xi + (k - 0.5) * width, m + 1.2, txt, ha="center",
                        va="bottom", fontsize=6.8)
                if m >= 1.0:
                    pk = peak_of(b, state, conds[int(xi)][0], window)
                    ax.text(xi + (k - 0.5) * width, 0.8,
                            f"{np.nanmedian(pk):.1f} Hz", ha="center",
                            va="bottom", fontsize=6.0, color="#444444")
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _c, lbl, _col in conds], fontsize=9)
        ax.set_ylabel("Population rate (Hz)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(handles=[Patch(facecolor="#444444", label="EO (gKv3=40)"),
                        Patch(facecolor="#888888", alpha=0.55, hatch="//",
                              label="EC (gKv3=0)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2,
               fontsize=9, frameon=False)

    # ---------------- (b) mode maps ---------------- #
    g_vals = sorted({float(r["g_gabab"]) for r in a2})
    amp_vals = sorted({float(r["amp_tacs"]) for r in a2})
    cmap = ListedColormap([c for _n, c, _t, _w in MODES])
    norm = BoundaryNorm(np.arange(-0.5, len(MODES) + 0.5), len(MODES))

    axes_b = [fig.add_axes([0.055, 0.07, 0.38, 0.36]),
              fig.add_axes([0.545, 0.07, 0.38, 0.36])]
    for ax, state in zip(axes_b, STATES):
        mat = np.zeros((len(g_vals), len(amp_vals)))
        txt = np.empty_like(mat, dtype=object)
        for i, g in enumerate(g_vals):
            for j, amp in enumerate(amp_vals):
                sel = [r for r in a2
                       if r["state"] == state and abs(float(r["g_gabab"]) - g) < 1e-9
                       and abs(float(r["amp_tacs"]) - amp) < 1e-9]
                if not sel:
                    mat[i, j] = MODE_INDEX["other"]
                    txt[i, j] = ""
                    continue
                modes = [classify(amp, float(r["rate"]), float(r["f_peak"]),
                                  float(r["prom"])) for r in sel]
                counts = {m: modes.count(m) for m in set(modes)}
                best = max(counts, key=counts.get)
                mat[i, j] = MODE_INDEX[best]
                txt[i, j] = dict((n, t) for n, _c, t, _w in MODES)[best]
                if len(counts) > 1:
                    txt[i, j] += "*"
        ax.imshow(mat, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for i in range(len(g_vals)):
            for j in range(len(amp_vals)):
                white = MODES[int(mat[i, j])][3]
                ax.text(j, i, txt[i, j], ha="center", va="center", fontsize=9,
                        weight="bold", color="white" if white else "black")
        ax.set_xticks(range(len(amp_vals)))
        ax.set_xticklabels([f"{a:g}" for a in amp_vals])
        ax.set_yticks(range(len(g_vals)))
        ax.set_yticklabels([f"{g:g}" for g in g_vals])
        ax.set_xlabel("tACS amplitude (µA/cm²)")
        ax.set_ylabel("g_GABA_B (mS/cm²)")
        ax.set_title(f"({'cd'[STATES.index(state)]}) locked mode, {state}",
                     loc="left", fontsize=11)

    axes_b[0].legend(handles=[Patch(facecolor=c, label=t)
                              for _n, c, t, _w in MODES if _n != "other"],
                     loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=6,
                     fontsize=8, frameon=False)

    out = os.path.join(FIG, "fig4_switch_v2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved: {out}\n")

    # ---------------- numbers for the caption ---------------- #
    print("Panel (a): mean +/- sd (n=10), fold change vs the SAME state's Sham")
    for state in STATES:
        sham = rate_of(b, state, "Sham", "iTBS").mean()
        sham2 = rate_of(b, state, "Sham", "tACS").mean()
        for cond, window, d in [("Sham", "iTBS", sham), ("iTBS", "iTBS", sham),
                                ("iTBS+tACS", "iTBS", sham),
                                ("Sham", "tACS", sham2), ("iTBS+tACS", "tACS", sham2),
                                ("tACS_only", "tACS", sham2)]:
            v = rate_of(b, state, cond, window)
            pk = peak_of(b, state, cond, window)
            print(f"  {state} | {cond:10s} | {window:5s} | {v.mean():6.2f} +/- {v.std():5.2f} Hz"
                  f" | x{v.mean() / d:5.2f} | f_peak {np.nanmedian(pk):5.2f} Hz")
    print("\nPanel (b): modes from raw rate/f_peak (amp=0 = intrinsic, no lock claim)")
    for state in STATES:
        for g in g_vals:
            line = []
            for amp in amp_vals:
                sel = [r for r in a2 if r["state"] == state
                       and abs(float(r["g_gabab"]) - g) < 1e-9
                       and abs(float(r["amp_tacs"]) - amp) < 1e-9]
                modes = [classify(amp, float(r["rate"]), float(r["f_peak"]),
                                  float(r["prom"])) for r in sel]
                tag = max(set(modes), key=modes.count) if modes else "-"
                line.append(f"{amp:g}:{tag}{'*' if len(set(modes)) > 1 else ''}")
            print(f"  {state} g={g:<6g} " + "  ".join(line))


if __name__ == "__main__":
    main()
