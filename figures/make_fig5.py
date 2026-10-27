"""WP-F: rebuild Figure 5.  No simulation.

Changes from v0.1:
  * title no longer says "STDP" (the rule is a uniform, clock-driven weight
    modulation, not spike-timing dependent) and drops the internal "(P2)" tag;
  * the baked-in "T2.7a/T2.7b" titles are cropped off the two source panels;
  * NEW panel (c): the matched 5 s power ratio for LTD / LTP / A=0 control,
    which is where the plasticity claim actually lives.  v0.1 had no such panel.

Output: figures/fig5_plasticity_v2.png
Usage:  python tools/make_fig5.py
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
RE_CSV = os.path.join(S2, "c_reanalysis.csv")
A_PNG = os.path.join(FIG, "f_t27a_stdp_concept.png")
B_PNG = os.path.join(FIG, "f_t27b_long_stdp.png")

ARMS = [("ltd", "LTD window (w↓)", "#2ca02c"),
        ("ltp", "LTP window (w↑)", "#9467bd"),
        ("ctrl", "A=0 control (w fixed)", "#9e9e9e")]


def main():
    rows = []
    with open(RE_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if abs(float(r["wlen_s"]) - 5.0) < 1e-9:
                rows.append(r)

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("Figure 5. Phase-gated inhibitory weight modulation shapes gamma "
                 "bidirectionally.", fontsize=14, y=0.985, weight="bold")

    ax_a = fig.add_axes([0.045, 0.50, 0.43, 0.40])
    ax_b = fig.add_axes([0.525, 0.50, 0.43, 0.40])
    for ax, path, title in [
        (ax_a, A_PNG, "(a) Conceptual validation (LTD window, 17.5 s)"),
        (ax_b, B_PNG, "(b) Long-term weight trajectory (60 s stim, 120 s post)"),
    ]:
        if os.path.exists(path):
            img = Image.open(path)
            w, h = img.size
            ax.imshow(np.asarray(img.crop((0, int(0.055 * h), w, h))))
            ax.axis("off")
        ax.set_title(title, loc="left", fontsize=11)

    # ---- (c) matched-window power ratios ----
    ax_c = fig.add_axes([0.07, 0.07, 0.86, 0.34])
    width = 0.34
    x = np.arange(len(ARMS))
    for k, window in enumerate(["early", "late"]):
        # median and IQR, not mean and SD: the paper's statistics section reports
        # median [IQR] because power ratios are heavy-tailed, so the figure must
        # use the same estimator or the caption and the text disagree.
        meds, lo_err, hi_err = [], [], []
        for arm, _lbl, _c in ARMS:
            base = {r["seed"]: float(r["power"]) for r in rows
                    if r["arm"] == arm and r["window"] == "base"}
            vals = np.array([float(r["power"]) / base[r["seed"]] for r in rows
                             if r["arm"] == arm and r["window"] == window
                             and r["seed"] in base and base[r["seed"]] > 0])
            med = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            meds.append(med)
            lo_err.append(med - q1)
            hi_err.append(q3 - med)
            jitter = np.linspace(-0.05, 0.05, len(vals)) if len(vals) > 1 else [0]
            ax_c.scatter(np.full(len(vals), x[ARMS.index((arm, _lbl, _c))]
                                 + (k - 0.5) * width) + jitter,
                         vals, s=28, color="#222222", alpha=0.55, zorder=4,
                         facecolor="none", linewidth=0.8)
        offset = (k - 0.5) * width
        ax_c.bar(x + offset, meds, width,
                 color=["#bfe3bf", "#d9c7e8"][k], edgecolor="#444444",
                 linewidth=0.6, label=["early (0–5 s)", "late (25–30 s)"][k])
        # IQR drawn explicitly (Q1-Q3) instead of bar(yerr=...): the whisker must
        # be visibly the quartile range, and the label must clear it.
        for xi, m, lo, hi in zip(x, meds, lo_err, hi_err):
            xpos = xi + offset
            ax_c.vlines(xpos, m - lo, m + hi, color="#333333", lw=1.4, zorder=5)
            ax_c.hlines([m - lo, m + hi], xpos - 0.06, xpos + 0.06,
                        color="#333333", lw=1.4, zorder=5)
            ax_c.text(xpos, m + hi + 0.035, f"{m:.2f}", ha="center",
                      va="bottom", fontsize=8)

    ax_c.axhline(1.0, color="#444444", ls="--", lw=1.0)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"{lbl}\n(n={sum(1 for r in rows if r['arm'] == arm and r['window'] == 'early')})"
                          for arm, lbl, _c in ARMS], fontsize=9)
    ax_c.set_ylabel("30–50 Hz power ratio, median [IQR]\n(post / baseline, matched 5 s windows)")
    ax_c.set_title("(c) matched-window gamma power after stimulation", loc="left",
                   fontsize=11)
    ax_c.legend(fontsize=9, frameon=False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    out = os.path.join(FIG, "fig5_plasticity_v2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved: {out}\n")

    for arm, lbl, _c in ARMS:
        base = {r["seed"]: float(r["power"]) for r in rows
                if r["arm"] == arm and r["window"] == "base"}
        for window in ["early", "late"]:
            vals = np.array([float(r["power"]) / base[r["seed"]] for r in rows
                             if r["arm"] == arm and r["window"] == window
                             and r["seed"] in base and base[r["seed"]] > 0])
            print(f"{arm:5s} {window:5s} n={len(vals):2d} "
                  f"median={np.median(vals):.2f} [{np.percentile(vals, 25):.2f}-"
                  f"{np.percentile(vals, 75):.2f}] mean={vals.mean():.2f}")


if __name__ == "__main__":
    main()
