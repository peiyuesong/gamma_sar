"""Compose Figure 5: Phase-gated STDP produces persistent gamma aftereffect (P2)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fig = plt.figure(figsize=(16, 7.5))

# 子图上提，压缩顶部空白
ax_a = fig.add_axes([0.04, 0.08, 0.45, 0.82])
img_a = mpimg.imread(os.path.join(base, "figures", "f_t27a_stdp_LTD.png"))
ax_a.imshow(img_a)
ax_a.axis("off")
ax_a.set_title("(a) Conceptual validation (17.5 s bio time)", fontsize=12, loc="left", pad=6)

ax_b = fig.add_axes([0.52, 0.08, 0.45, 0.82])
img_b = mpimg.imread(os.path.join(base, "figures", "f_t27b_long_stdp.png"))
ax_b.imshow(img_b)
ax_b.axis("off")
ax_b.set_title("(b) Long-term P2 aftereffect (60 s stim + 120 s post)", fontsize=12, loc="left", pad=6)

fig.suptitle(
    "Figure 5. Phase-gated STDP produces a persistent gamma aftereffect (P2).",
    fontsize=14, y=0.97, weight="bold"
)

out = os.path.join(base, "figures", "fig5_stdp_p2.png")
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")