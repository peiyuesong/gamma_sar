"""从 t13_fI_surface.csv 重新绘制全英文的 f-I 图（解决中文字体乱码）。"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(HERE, "t13_fI_surface.csv")
FIG_PATH = os.path.join(ROOT, "figures", "f1_fI_surface.png")


def main():
    data = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data.append(r)

    G_LIST = sorted(set(float(r["g_kv3_mScm2"]) for r in data))
    I_LIST = sorted(set(float(r["i_uA_cm2"]) for r in data))

    rates = {g: [] for g in G_LIST}
    vends = {g: [] for g in G_LIST}
    blocks = {g: [] for g in G_LIST}

    for g in G_LIST:
        for i in I_LIST:
            row = next(r for r in data
                       if float(r["g_kv3_mScm2"]) == g
                       and float(r["i_uA_cm2"]) == i)
            rates[g].append(float(row["rate_hz"]))
            vends[g].append(float(row["v_end_mV"]))
            blocks[g].append(int(row["blocked"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(G_LIST)))

    for idx, g in enumerate(G_LIST):
        lbl = f"g_Kv3 = {g:.0f}"
        axes[0].plot(I_LIST, rates[g], "-o", color=cmap[idx], ms=3, label=lbl)
        axes[1].plot(I_LIST, vends[g], "-o", color=cmap[idx], ms=3, label=lbl)
        for c in range(len(I_LIST)):
            if blocks[g][c]:
                axes[0].plot(I_LIST[c], rates[g][c], "x", color="red", ms=7)
                axes[1].plot(I_LIST[c], vends[g][c], "x", color="red", ms=7)

    axes[0].set(xlabel="Injected Current ($\\mu$A/cm$^2$)",
                ylabel="Firing Rate (Hz)",
                title="f-I Surface")
    axes[1].axhline(-35, color="red", ls=":", lw=1)
    axes[1].set(xlabel="Injected Current ($\\mu$A/cm$^2$)",
                ylabel="Steady-state $v_{end}$ (mV)",
                title="Steady-state $v_{end}$ (red X = BLOCK)")
    axes[1].legend(fontsize=8, ncol=2)

    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved: {FIG_PATH}")


if __name__ == "__main__":
    main()