"""Compose Figure 2: ING network and GABA_B-mediated silencing."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fig = plt.figure(figsize=(16, 6.8))

ax_a = fig.add_axes([0.04, 0.08, 0.44, 0.80])
img_a = mpimg.imread(os.path.join(base, "figures", "f_t24a_gabab_ipsp.png"))
ax_a.imshow(img_a)
ax_a.axis("off")

ax_b = fig.add_axes([0.52, 0.08, 0.44, 0.80])
img_b = mpimg.imread(os.path.join(base, "figures", "f_t24b_gabab_network.png"))
ax_b.imshow(img_b)
ax_b.axis("off")

fig.text(0.04, 0.92, "(a) Single-cell GABA_B IPSP (IPSP = 22.7 mV, 1 s duration)",
         fontsize=11, ha="left", va="top", weight="bold")
fig.text(0.52, 0.92, "(b) Network silencing via GABA_B (rate drop = 90.0%)",
         fontsize=11, ha="left", va="top", weight="bold")

fig.suptitle(
    "Figure 2. GABA_B activation hyperpolarizes PV interneurons and silences the ING network.",
    fontsize=13, y=0.97, weight="bold"
)

out = os.path.join(base, "figures", "fig2_gabab_silencing.png")
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")