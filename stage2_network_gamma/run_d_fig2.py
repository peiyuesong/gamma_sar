"""WP-D: rebuild Figure 2 on ONE implementation (Destexhe network GABA_B).

Why this exists
---------------
v0.1's Figure 2 mixed three incompatible protocols:
  panel (b) representative trace : Destexhe gating, g=0.05, RECTANGULAR pulse, I=0.85
  panel (c) 10-seed bars         : constant -0.125 uA/cm2 current, I=0.80
  the intervention (Fig 4)       : Destexhe gating, g=0.005, 20 ms pulses, I=0.85
so the "same GABA_B pulse" in the text referred to three different things, and the
"90.0%" in the abstract came from the constant-current version.

This script rebuilds panels (b) and (c) on the frozen protocol:
  I_mu = 0.85, g_GABA_B = 0.02, 20 ms pulses every 100 ms for 2 s (1000-3000 ms)
  (a) stays as the existing single-cell IPSP panel -- that measurement is
      independent of the network dose and is not part of the inconsistency.

Panel (c) is NOT re-simulated: it is the iTBS arm of b_stats.csv (10 seeds, same
protocol), so one experiment feeds both Fig 2c and Fig 4.

Outputs
-------
  figures/fig2_gabab_silencing_v2.png
  stage2_network_gamma/fig2_spikes/fig2_repr_seed0.npz
  stdout: the exact numbers that go into the caption

Usage:  python stage2_network_gamma/run_d_fig2.py
"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_a1_pilot import run_sim, N, I_MU, DT_MS, ITBS_PROTOCOL
from common.params import PARAMS

EO_KV3 = 40.0
G_GABAB = 0.02
SEED = 0
T_END = 9000.0
BIN_MS = 100.0

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(OUT_DIR), "figures")
B_CSV = os.path.join(OUT_DIR, "b_stats.csv")


def load_b_stats():
    """Pull the EO iTBS arm: baseline / silenced / recovered, n=10."""
    base, sil, rec = [], [], []
    with open(B_CSV, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["state"] != "EO":
                continue
            r = float(row["rate"])
            if row["condition"] == "Sham" and row["window"] == "iTBS":
                base.append(r)
            elif row["condition"] == "iTBS" and row["window"] == "iTBS":
                sil.append(r)
            elif row["condition"] == "iTBS" and row["window"] == "post":
                rec.append(r)
    return np.array(base), np.array(sil), np.array(rec)


def main():
    print("=" * 88)
    print(f"WP-D Figure 2 rebuild | I_mu={I_MU}, g_GABA_B={G_GABAB}, "
          f"20 ms/100 ms x 2 s, dt={DT_MS} ms")
    print("=" * 88)

    # ---- representative run (panel b) ----
    t_sp = run_sim(EO_KV3, "iTBS", SEED, g_gabab=G_GABAB, amp_tacs=0.0, t_end=T_END)
    sp_dir = os.path.join(OUT_DIR, "fig2_spikes")
    os.makedirs(sp_dir, exist_ok=True)
    np.savez_compressed(os.path.join(sp_dir, f"fig2_repr_seed{SEED}.npz"),
                        t_ms=t_sp.astype(np.float32), g_gabab=np.array(G_GABAB),
                        seed=np.array(SEED), i_mu=np.array(I_MU))

    bins = np.arange(0.0, T_END + BIN_MS, BIN_MS)
    hist, _ = np.histogram(t_sp, bins=bins)
    rate = hist / (N * BIN_MS / 1000.0)
    centers = (bins[:-1] + bins[1:]) / 2.0

    # ---- 10-seed stats (panel c) from b_stats.csv ----
    base, sil, rec = load_b_stats()
    if not (len(base) == len(sil) == len(rec) == 10):
        raise RuntimeError(f"expected 10 seeds per arm, got "
                           f"{len(base)}/{len(sil)}/{len(rec)}")
    supp = (1 - sil.mean() / base.mean()) * 100
    print(f"baseline   {base.mean():6.2f} +/- {base.std():5.2f} Hz  (n={len(base)})")
    print(f"iTBS 1-3 s {sil.mean():6.2f} +/- {sil.std():5.2f} Hz  "
          f"-> {supp:.1f}% suppression")
    print(f"recovered  {rec.mean():6.2f} +/- {rec.std():5.2f} Hz")
    print(f"representative seed {SEED}: min bin rate {rate.min():.2f} Hz")
    print("\nretired v0.1 numbers (NOT to be cited): 90.0% suppression, "
          "constant current, I=0.80")

    # ---- compose ----
    fig = plt.figure(figsize=(17, 5.2))
    fig.suptitle("Figure 2. GABA_B activation hyperpolarises PV interneurons and "
                 "silences the ING network.", fontsize=13, y=0.99, weight="bold")

    # (a) existing single-cell IPSP panel
    ax_a = fig.add_axes([0.03, 0.10, 0.28, 0.72])
    img_path = os.path.join(FIG_DIR, "f_t24a_gabab_ipsp.png")
    # the single-cell demonstration uses PARAMS["gabab"].g_GABA_B (1.0 mS/cm2),
    # NOT the network's working dose (0.02) -- label it so the two panels cannot
    # be read as the same conductance.
    g_demo = PARAMS["gabab"].g_GABA_B
    if os.path.exists(img_path):
        # the source PNG has "[PASS]" and a T2.x-style label baked into its top
        # strip; crop that strip and write the publication title here instead
        from PIL import Image
        img = Image.open(img_path)
        w, h = img.size
        ax_a.imshow(np.asarray(img.crop((0, int(0.055 * h), w, h))))
        ax_a.axis("off")
    else:
        ax_a.text(0.5, 0.5, "single-cell IPSP panel missing",
                  ha="center", va="center")
        ax_a.axis("off")
    ax_a.set_title(f"(a) Single-cell GABA_B IPSP (g={g_demo:g})\n"
                   f"IPSP = 22.7 mV, duration 1000 ms",
                   loc="left", fontsize=10)

    # (b) representative rate trace
    ax_b = fig.add_axes([0.36, 0.16, 0.28, 0.66])
    ax_b.axvspan(1.0, 3.0, color="#d62728", alpha=0.12, lw=0)
    ax_b.plot(centers / 1000.0, rate, color="black", lw=0.9)
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Population rate (Hz)")
    ax_b.set_xlim(0, T_END / 1000.0)
    ax_b.set_ylim(bottom=0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.set_title(f"(b) representative seed (g={G_GABAB})", loc="left", fontsize=11)

    # (c) 10-seed bars
    ax_c = fig.add_axes([0.70, 0.16, 0.27, 0.66])
    means = [base.mean(), sil.mean(), rec.mean()]
    sds = [base.std(), sil.std(), rec.std()]
    labels = ["Baseline\n(Sham)", "iTBS\n(1-3 s)", "Recovered\n(7-9 s)"]
    bars = ax_c.bar(labels, means, yerr=sds, capsize=4, width=0.55,
                    color=["#1f77b4", "#d62728", "#2ca02c"],
                    edgecolor="white", linewidth=0.8)
    for bar, m, s in zip(bars, means, sds):
        ax_c.text(bar.get_x() + bar.get_width() / 2, m + s + 0.4,
                  f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=9)
    ax_c.set_ylabel("Population rate (Hz)")
    ax_c.set_ylim(0, max(means) + 4)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.set_title(f"(c) {len(base)} seeds, EO", loc="left", fontsize=11)
    # No percentage text inside the figure: the suppression figure lives in the
    # caption and in the text, not on the axes.  Bar labels are raw data only.

    out = os.path.join(FIG_DIR, "fig2_gabab_silencing_v2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nsaved figure: {out}")
    print(f"saved spikes: {sp_dir}")


if __name__ == "__main__":
    main()
