"""T2.1d 绘图：读 t21d_fine.csv -> figures/f2_operating_map.png。

含分类卫生修正（v2）：真 gamma 要求 rate >= 8 Hz 且 sync_r >= 0.1，
低驱动的零星放电（rate<8 的假峰）改判 async。下沿读数以修正后为准。
"""
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import csv
from collections import defaultdict


def load():
    data = defaultdict(list)
    with io.open(os.path.join(HERE, "t21d_fine.csv"), encoding="utf-8") as f:
        for r in csv.reader(f):
            if not r or r[0] == "g_kv3":
                continue
            g, k, i, sd = float(r[0]), float(r[1]), float(r[2]), int(r[3])
            rate, f0, sr = float(r[4]), float(r[5]), float(r[6])
            data[(g, k, i)].append((rate, f0, sr))
    return data


def reclass(rate, f0, sr):
    """分类卫生：gamma 需要 rate>=8 且 sync>=0.1。"""
    if rate < 8.0 or sr < 0.10:
        return "async"
    if abs(f0 / rate - 1.0) < 0.25:
        return "sync-1:1"
    if f0 / rate > 1.5:
        return "ING-2:1"
    return "sync?"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    data = load()
    Ivals = sorted({i for (g, k, i) in data if k == 20.0})
    colors = {0.0: "#1f77b4", 40.0: "#d62728"}
    labels = {0.0: "gKv3=0 (EC-axis)", 40.0: "gKv3=40 (EO-anchor)"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    print(f"{'i':>5} | {'gKv3=0 2:1%[CI]':>16} {'f0_21':>6} | "
          f"{'gKv3=40 2:1%[CI]':>16} {'f0_21':>6}")
    for g in (0.0, 40.0):
        f21m, f11m, fr = [], [], []
        for i in Ivals:
            rs = [(r[0], r[1], r[2], reclass(*r)) for r in data[(g, 20.0, i)]]
            n = len(rs)
            k21 = sum(1 for r in rs if r[3] == "ING-2:1")
            p, lo, hi = wilson(k21, n)
            fr.append((p, lo, hi))
            a = [r[1] for r in rs if r[3] == "ING-2:1"]
            b = [r[1] for r in rs if r[3] == "sync-1:1"]
            f21m.append(np.median(a) if a else np.nan)
            f11m.append(np.median(b) if b else np.nan)
            if g == 0.0:
                print(f"{i:>5.2f} | {p:>5.0%}[{lo:>4.0%},{hi:>4.0%}] "
                      f"{(np.median(a) if a else 0):>6.1f} |", end=" ")
            else:
                print(f"{p:>5.0%}[{lo:>4.0%},{hi:>4.0%}] "
                      f"{(np.median(a) if a else 0):>6.1f}")
        ax1.plot(Ivals, f21m, "o-", color=colors[g], label=labels[g] + " 2:1")
        ax1.plot(Ivals, f11m, "s--", color=colors[g], alpha=0.5,
                 label=labels[g] + " 1:1")
        fr = np.array(fr)
        ax2.plot(Ivals, fr[:, 0], "o-", color=colors[g], label=labels[g])
        ax2.fill_between(Ivals, fr[:, 1], fr[:, 2], color=colors[g], alpha=0.2)

    ax1.axhspan(38, 42, color="green", alpha=0.12)
    ax1.text(1.19, 40, "tACS 40Hz", color="green", fontsize=9, va="center")
    ax1.set_ylabel("f0 (Hz)"); ax1.set_ylim(15, 68)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("T2.1d operating map, hygiene-corrected (N=100, 20 seeds)")
    ax2.axhline(0.5, color="gray", ls=":", lw=1)
    ax2.set_ylabel("P(2:1 | i, gKv3)"); ax2.set_xlabel("drive i (uA/cm2)")
    ax2.set_ylim(-0.03, 1.08); ax2.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    out = os.path.join(ROOT, "figures", "f2_t21d_operating_map.png")
    plt.savefig(out, dpi=150)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()