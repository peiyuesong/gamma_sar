"""Compose Figure 1: Single-cell validation from MS1."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

img_a = mpimg.imread(os.path.join(base, "figures", "f1_fI_surface.png"))
img_b = mpimg.imread(os.path.join(base, "figures", "f2_t21d_operating_map.png"))
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]
ratio_a = w_a / h_a
ratio_b = w_b / h_b

fig_w, fig_h = 14.0, 5.0
side = 0.02       # 边距压缩到 2%
gap = 0.02        # 间隙压缩到 2%
bottom = 0.05

avail_w = fig_w * (1 - 2*side - gap)
disp_h = avail_w / (ratio_a + ratio_b)

w_a = (disp_h * ratio_a) / fig_w
w_b = (disp_h * ratio_b) / fig_w
h = disp_h / fig_h

left_a = side
left_b = left_a + w_a + gap

fig = plt.figure(figsize=(fig_w, fig_h))

ax_a = fig.add_axes([left_a, bottom, w_a, h])
ax_a.imshow(img_a)
ax_a.axis("off")

ax_b = fig.add_axes([left_b, bottom, w_b, h])
ax_b.imshow(img_b)
ax_b.axis("off")

fig.text(left_a, bottom + h + 0.015,
         "(a) f-I surface (MS1 Hodgkin-Huxley validation)",
         fontsize=11, ha="left", va="bottom", weight="bold")
fig.text(left_b, bottom + h + 0.015,
         "(b) Operating map (2:1 vs 1:1 mode, hygiene-corrected)",
         fontsize=11, ha="left", va="bottom", weight="bold")

fig.suptitle(
    "Figure 1. Single-cell model validation and ING operating regime.",
    fontsize=13, y=0.97, weight="bold"
)

out = os.path.join(base, "figures", "fig1_single_cell_validation.png")
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")