"""Compose Fig 2 v3: equal-size panels, trimmed embeds, perfectly aligned."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def trim_border(img, threshold=250):
    """Crop uniform border (white or near-white) from PIL Image."""
    img_rgb = img.convert("RGB") if img.mode != "RGB" else img
    arr = np.array(img_rgb)
    mask = np.any(arr < threshold, axis=2)
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img
    top = int(np.argmax(rows))
    bottom = int(len(rows) - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(len(cols) - np.argmax(cols[::-1]))
    return img.crop((left, top, right, bottom))


fig_dir = os.path.dirname(os.path.abspath(__file__))

img_a = trim_border(Image.open(os.path.join(fig_dir, "f_t24a_gabab_ipsp.png")))
img_b = trim_border(Image.open(os.path.join(fig_dir, "f_t24b_gabab_network.png")))

fig = plt.figure(figsize=(17, 5.5))

fig.text(0.5, 0.97,
         "Figure 2. GABA_B activation hyperpolarises PV interneurons and suppresses the ING network.",
         ha="center", va="top", fontsize=13, weight="bold")

# Three panels: EQUAL width and height, perfectly aligned
h = 0.72
y0 = 0.10
w = 0.28
gap = 0.015

# --- panel (a) ---
ax_a = fig.add_axes([0.03, y0, w, h])
ax_a.imshow(img_a)
ax_a.axis("off")
ax_a.set_title("(a) Single-cell GABA_B IPSP", loc="left", fontsize=11, y=1.02)

# --- panel (b) ---
ax_b = fig.add_axes([0.03 + w + gap, y0, w, h])
ax_b.imshow(img_b)
ax_b.axis("off")
ax_b.set_title("(b) Network silencing (representative)", loc="left", fontsize=11, y=1.02)

# --- panel (c) native bar chart ---
ax_c = fig.add_axes([0.03 + 2*(w + gap), y0, w, h])
conditions = ["Baseline", "GABA_B (2s)", "Post"]
means = [12.67, 1.27, 13.24]
stds = [2.26, 0.47, 2.71]
colors = ["#1f77b4", "#d62728", "#2ca02c"]

bars = ax_c.bar(conditions, means, yerr=stds, capsize=4, color=colors,
                edgecolor="white", linewidth=0.8, width=0.55)
ax_c.set_ylabel("Population rate (Hz)", fontsize=10.5)
ax_c.set_ylim(0, 17)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.tick_params(axis="both", labelsize=9.5)

for bar, m, s in zip(bars, means, stds):
    ax_c.text(bar.get_x() + bar.get_width()/2, m + s + 0.35,
              f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=9)

ax_c.annotate("90.0% reduction", xy=(1, 0), xytext=(1, -2.6),
              textcoords="data", ha="center", va="top",
              fontsize=9, color="#d62728", weight="bold")

ax_c.text(0.03, 0.03, "paired t(9)=17.4\np=3.15×10⁻⁸\nCohen's d=6.63",
          transform=ax_c.transAxes, ha="left", va="bottom", fontsize=8,
          bbox=dict(boxstyle="round,pad=0.25", facecolor="wheat", alpha=0.45))

ax_c.set_title("(c) 10-seed validation", loc="left", fontsize=11, y=1.02)

plt.savefig(os.path.join(fig_dir, "fig2_gabab_silencing.png"),
            dpi=300, bbox_inches="tight")
print("Saved: fig2_gabab_silencing.png")
