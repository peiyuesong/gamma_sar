"""Compose Figure 3: tACS 40Hz entrainment dose-response."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fig = plt.figure(figsize=(16, 6.2))

ax_a = fig.add_axes([0.04, 0.06, 0.45, 0.86])
img_a = mpimg.imread(os.path.join(base, "figures", "f_t25b_tacs_entrainment.png"))
ax_a.imshow(img_a, aspect="auto")
ax_a.axis("off")

ax_b = fig.add_axes([0.52, 0.06, 0.45, 0.86])
img_b = mpimg.imread(os.path.join(base, "figures", "f_t25c_tacs_extended.png"))
ax_b.imshow(img_b, aspect="auto")
ax_b.axis("off")

# (a)(b) 放在 figure 层面，位于 suptitle 下方、子图上方
fig.text(0.04, 0.935, "(a) Fine dose-response (0–2.5 µA/cm², 13 pts)",
         fontsize=11, ha="left", va="top", weight="bold")
fig.text(0.52, 0.935, "(b) Extended range (0.05–10.0 µA/cm², 17 pts)",
         fontsize=11, ha="left", va="top", weight="bold")

fig.suptitle(
    "Figure 3. tACS 40Hz entrainment dose-response reveals four distinct regimes.",
    fontsize=13, y=0.97, weight="bold"
)

out = os.path.join(base, "figures", "fig3_tacs_dose_response.png")
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")