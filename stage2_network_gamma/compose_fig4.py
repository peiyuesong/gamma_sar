"""Compose Figure 4: Combined intervention switch."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

img_a = mpimg.imread(os.path.join(base, "figures", "f_t26_eo_ec_reversal.png"))
img_b = mpimg.imread(os.path.join(base, "figures", "f_t26c_param_scan.png"))
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]
ratio_a = w_a / h_a
ratio_b = w_b / h_b

fig_w, fig_h = 14.0, 4.2        # 收窄画布
disp_h = 3.0                    # 图更大
h = disp_h / fig_h
w_a = (disp_h * ratio_a) / fig_w
w_b = (disp_h * ratio_b) / fig_w

bottom = 0.06
left_a = 0.03
left_b = left_a + w_a + 0.04

fig = plt.figure(figsize=(fig_w, fig_h))

ax_a = fig.add_axes([left_a, bottom, w_a, h])
ax_a.imshow(img_a)
ax_a.axis("off")

ax_b = fig.add_axes([left_b, bottom, w_b, h])
ax_b.imshow(img_b)
ax_b.axis("off")

fig.text(left_a, bottom + h + 0.015,
         "(a) Intervention bar chart (Sham / iTBS / tACS+iTBS)",
         fontsize=11, ha="left", va="bottom", weight="bold")
fig.text(left_b, bottom + h + 0.015,
         "(b) Parameter heatmap (GABA_B × tACS, EO vs EC Kv3)",
         fontsize=11, ha="left", va="bottom", weight="bold")

fig.suptitle(
    "Figure 4. Combined intervention switch: iTBS closes gamma, tACS reopens it.",
    fontsize=13, y=0.98, weight="bold"
)

out = os.path.join(base, "figures", "fig4_combined_switch.png")
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")