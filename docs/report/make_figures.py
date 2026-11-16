"""Create vector figures from external development aggregates and synthetic examples."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "external_metrics.json"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white",
})
INK = "#24313b"
BLUE = "#126774"
ORANGE = "#be5c30"
GREY = "#657180"

def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", pad_inches=.045)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(7.1, 1.95))
fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
stages = [
    ("Tracking features", "Ball + team movement\nMissingness; 5 Hz"),
    ("Recurrent ensemble", "4 components × 3 seeds\n2 heads: dead / control"),
    ("Symmetric inference", "Original + team swap\nAverage log-odds"),
    ("Sequence and CSV", "Joint H / A / D decoder\n199,319 frame labels"),
]
for i, (title, detail) in enumerate(stages):
    x = .01 + i * .25
    ax.add_patch(FancyBboxPatch((x, .52), .22, .42,
        boxstyle="round,pad=.006,rounding_size=.018", linewidth=.8,
        edgecolor="#a8bec5", facecolor="#eef5f6"))
    ax.text(x+.11, .82, title, ha="center", va="center", color=BLUE, fontsize=9, weight="bold")
    ax.text(x+.11, .66, detail, ha="center", va="center", color=INK, fontsize=8.5, linespacing=1.5)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x+.226, .72), (x+.244, .72), arrowstyle="-|>",
                                    mutation_scale=8, linewidth=.8, color=GREY))
ax.add_patch(FancyBboxPatch((.01, .035), .48, .27,
    boxstyle="round,pad=.008,rounding_size=.018", linewidth=.8,
    edgecolor="#c8c5b9", facecolor="#f8f5ed"))
ax.text(.25, .24, "External training", ha="center", color=INK, fontsize=9, weight="bold")
ax.text(.25, .13, "7 DFL + 20 SkillCorner matches\n+9 older SkillCorner: one control component",
        ha="center", va="center", color=INK, fontsize=8.5, linespacing=1.4)
ax.add_patch(FancyArrowPatch((.37, .32), (.37, .51), arrowstyle="-|>",
                            mutation_scale=9, linewidth=.9, color=GREY))
ax.text(.75, .18, "Per-half decoding · switch costs 50/50\nNo fixed D override · X outside scored periods",
        ha="center", va="center", fontsize=8.3, color=GREY, linespacing=1.7)
save(fig, "pipeline")

m = json.loads(SOURCE.read_text(encoding="utf-8"))
conditions = ("deg", "clean", "ball_long_gaps_40")
references = ("raw", "2s", "3s")
values = np.array([[m["robustness"][k]["new"][r] - m["robustness"][k]["old"][r]
                    for r in references] for k in conditions])
cmap = LinearSegmentedColormap.from_list("signed_effect", [ORANGE, "#fafafa", BLUE])
norm = TwoSlopeNorm(vmin=-.06, vcenter=0, vmax=.06)
fig, ax = plt.subplots(figsize=(3.45, 2.55))
fig.subplots_adjust(left=.265, right=.97, top=.80, bottom=.31)
im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(range(3), ["Raw DFL", "2 s", "3 s"], fontsize=8.5)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_xlabel("Evaluation reference", fontsize=9, labelpad=9, color=INK)
ax.set_yticks(range(3), ["Degraded", "Clean", "40% gaps"], fontsize=8.5)
ax.tick_params(length=0, pad=6)
for i in range(3):
    for j in range(3):
        rgb = np.array(cmap(norm(values[i, j]))[:3])
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        luminance = linear @ np.array([.2126, .7152, .0722])
        black = (luminance + .05) / .05
        white = 1.05 / (luminance + .05)
        ax.text(j, i, f"{values[i,j]:+.4f}", ha="center", va="center",
                fontsize=8.7, weight="bold", color="white" if white > black else "black")
ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)
for spine in ax.spines.values(): spine.set_visible(False)
cax = fig.add_axes([.30, .15, .62, .045])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-.06, 0, .06])
cb.ax.tick_params(labelsize=7.5, length=2)
cb.set_label("2 s training minus 3 s training (spellF1)", fontsize=8, labelpad=4)
save(fig, "reference_tradeoff")

fig, ax = plt.subplots(figsize=(3.45, 1.85))
fig.subplots_adjust(left=.24, right=.98, top=.79, bottom=.27)
palette = {"H": BLUE, "A": ORANGE, "D": "#7e8790"}
rows = [
    (1, [(0,4,"H"), (4,5,"A"), (5,8,"H"), (8,10,"D"), (10,14,"A")]),
    (0, [(0,8,"H"), (8,10,"D"), (10,14,"A")]),
]
for y, intervals in rows:
    for left, right, label in intervals:
        ax.add_patch(Rectangle((left, y-.23), right-left, .46, facecolor=palette[label],
                               edgecolor="white", linewidth=.8))
        ax.text((left+right)/2, y, label, ha="center", va="center",
                fontsize=8, weight="bold", color="white")
ax.annotate("1 s switch", xy=(4.5, 1.23), xytext=(4.5, 1.64),
            ha="center", fontsize=8, color=INK,
            arrowprops={"arrowstyle":"->", "color":GREY, "lw":.7})
ax.set(xlim=(0,14), ylim=(-.38,1.84))
ax.set_yticks([1,0], ["Raw labels", "2 s reference"], fontsize=8)
ax.set_xticks([0,4,5,8,10,14])
ax.tick_params(axis="x", labelsize=7.5, length=2)
ax.tick_params(axis="y", length=0, pad=5)
ax.set_xlabel("Synthetic time (seconds)", fontsize=8, labelpad=4)
for spine in ("left","right","top"): ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#bcc5cb")
save(fig, "training_reference")

records = []
for c, row in zip(conditions, values):
    for r, delta in zip(references, row):
        records.append({"input":c,"reference":r,
                        "three_second_training":m["robustness"][c]["old"][r],
                        "two_second_training":m["robustness"][c]["new"][r],
                        "delta":float(delta)})
proof = {"scope":"External development match means only; no supplied match data or predictions.",
         "source":"external_metrics.json",
         "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         "dfl_matches":7,"metric":"spellF1","difference":"two-second minus three-second training",
         "synthetic_example":"H:0-4, A:4-5, H:5-8, D:8-10, A:10-14; filtered H:0-8, D:8-10, A:10-14",
         "rows":records}
(ROOT/"figure_data.json").write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"figures":["pipeline","reference_tradeoff","training_reference"],
                  "external_cells":len(records),"source_sha256":proof["source_sha256"]}))

