"""Compose Fig 4 v7: aligned panels, trimmed heatmap, aspect-matched."""
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
img_b = trim_border(Image.open(os.path.join(fig_dir, "f_t26c_param_scan.png")))

fig = plt.figure(figsize=(17, 5.5))

fig.text(0.5, 0.97,
         "Figure 4. Combined intervention switch: iTBS closes gamma, tACS reopens it.",
         ha="center", va="top", fontsize=13, weight="bold")

# Both panels share same bottom and height for perfect vertical alignment
h = 0.72
y0 = 0.10

# --- panel (a) native grouped bar chart ---
w_a = 0.30
ax_a = fig.add_axes([0.05, y0, w_a, h])
labels = ["Sham", "iTBS", "tACS+iTBS"]
x = np.arange(len(labels))
width = 0.32

eo_means = [67.6, 8.4, 233.2]
eo_stds = [8.6, 6.2, 29.7]
ec_means = [20.8, 1.4, 74.3]
ec_stds = [11.5, 1.1, 45.2]

bars1 = ax_a.bar(x - width/2, eo_means, width, yerr=eo_stds, capsize=4,
                 color="#1f77b4", label="EO (gKv3=40)", edgecolor="white", linewidth=0.6)
bars2 = ax_a.bar(x + width/2, ec_means, width, yerr=ec_stds, capsize=4,
                 color="#ff7f0e", label="EC (gKv3=20)", edgecolor="white", linewidth=0.6)

ax_a.set_ylabel("Stimulus-period gamma power (a.u.)", fontsize=11)
ax_a.set_xticks(x)
ax_a.set_xticklabels(labels, fontsize=10)
ax_a.legend(loc="upper left", frameon=False, fontsize=9)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.tick_params(axis="y", labelsize=10)
ax_a.set_ylim(0, 290)

ax_a.annotate("3.45×", xy=(x[2] - width/2, eo_means[2] + eo_stds[2] + 6),
              ha="center", va="bottom", fontsize=10, color="#1f77b4", weight="bold")
ax_a.annotate("3.57×", xy=(x[2] + width/2, ec_means[2] + ec_stds[2] + 6),
              ha="center", va="bottom", fontsize=10, color="#ff7f0e", weight="bold")

ax_a.set_title("(a) 10-seed intervention comparison (EO vs EC)", loc="left", fontsize=11, y=1.02)

# --- panel (b) embed heatmap, aspect-matched to source ratio ---
# Source f_t26c_param_scan.png = 1500 x 600 -> ratio 2.50 : 1
# Target w = 2.50 * h * fig_height / fig_width = 2.50 * 0.72 * 5.5 / 17 = 0.582
w_b = 0.58
gap = 0.03
ax_b = fig.add_axes([0.05 + w_a + gap, y0, w_b, h])
ax_b.imshow(img_b)
ax_b.axis("off")
ax_b.set_title("(b) Parameter heatmap (GABA_B × tACS)", loc="left", fontsize=11, y=1.02)

plt.savefig(os.path.join(fig_dir, "fig4_combined_switch.png"),
            dpi=300, bbox_inches="tight")
print("Saved: fig4_combined_switch.png")
