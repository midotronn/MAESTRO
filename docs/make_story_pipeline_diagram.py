"""Box-and-arrow diagram of the AgentLODGE LLM *story* (long-horizon) choreography pipeline.

Same visual language as make_pipeline_diagram.py (rounded colored nodes, bold title + subtitle,
curved black arrows, dashed side channels), but depicts the structure-aware story flow:
musical-form analysis -> LLM storyboard -> structure-aware assembly -> render -> comparison video.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10.5, 12.0))
ax.set_xlim(0, 100)
ax.set_ylim(6, 100)
ax.axis("off")

C = {
    "in": "#5B8FF9", "pre": "#61DDAA", "lodge": "#F6BD16", "edge": "#F08BB4",
    "struct": "#7DD3C0", "agent": "#FF9D4D", "asm": "#9270CA", "dance": "#5AD8A6",
    "vid": "#65789B", "cos": "#E8684A",
}
SIDE_C = "#C0392B"


def box(x, y, w, h, title, sub="", color="#ddd", tc="black", fs=11, subfs=8.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
        linewidth=1.4, edgecolor="#333", facecolor=color, alpha=0.93, zorder=2))
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.62, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
        ax.text(cx, y + h * 0.28, sub, ha="center", va="center",
                fontsize=subfs, color=tc, zorder=3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
    return (x, y, w, h)


def arr(p1, p2, color="#333", lw=1.9, style="-", rad=0.0, label="", labfs=7.5):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16, linewidth=lw, color=color,
        linestyle=style, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, label, ha="center",
                va="center", fontsize=labfs, style="italic", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.9), zorder=4)


def ortho(pts, color, lw=1.8, style="--", label="", label_pt=None, rot=90):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
            linestyle=style, lw=lw, zorder=1, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=style, mutation_scale=16), zorder=1)
    if label and label_pt:
        ax.text(*label_pt, label, ha="center", va="center", fontsize=7.5,
                style="italic", color=color, rotation=rot,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.9), zorder=4)


def bc(b):   # bottom-center
    return (b[0] + b[2] / 2, b[1])


def tc_(b):  # top-center
    return (b[0] + b[2] / 2, b[1] + b[3])


def lc(b):   # left-center
    return (b[0], b[1] + b[3] / 2)


def rc(b):   # right-center
    return (b[0] + b[2], b[1] + b[3] / 2)


ax.text(50, 97.5, "AgentLODGE, LLM Story Choreography", ha="center",
        va="center", fontsize=16, fontweight="bold")

b_song = box(38, 90, 24, 5.2, "Song", "audio in", color=C["in"], tc="white")
b_pre = box(26, 80, 48, 7, "Audio Preprocessing",
            "beats \u00b7 tempo \u00b7 features", color=C["pre"])

# generation branches + musical-form analysis (three across)
b_lodge = box(5, 66.5, 26, 7.5, "LODGE", "diffusion \u2192 dance", color=C["lodge"])
b_struct = box(36.5, 66.5, 27, 7.5, "Musical-Form Analysis",
               "sections + energy arc", color=C["struct"])
b_edge = box(69, 66.5, 26, 7.5, "EDGE", "long-form \u2192 dance", color=C["edge"])

b_agent = box(28, 54.5, 44, 8, "LLM Storyboard Agent",
              "arc + per-section plan (role, intensity, bias)",
              color=C["agent"], tc="white")
b_asm = box(24, 42.5, 52, 8, "Structure-Aware Assembly",
            "per-section select (coherence + arc + bias) \u00b7 inertial seams",
            color=C["asm"], tc="white", subfs=8.0)
b_dance = box(34, 33.5, 32, 5.4, "Story Dance",
              "LODGE\u2192EDGE\u2192LODGE", color=C["dance"], fs=10, subfs=8.0)
b_render = box(30, 22, 40, 7.8, "Blender Y-Bot Render",
               "3 panels \u00b7 480\u00d7480", color=C["vid"], tc="white")
b_video = box(24, 9.5, 52, 8.2, "Comparison Video",
              "timeline choices + LLM reasoning overlay",
              color=C["cos"], tc="white")

# main spine
arr(bc(b_song), tc_(b_pre))
arr((40, 80), tc_(b_lodge), rad=0.12)
arr(bc(b_pre), tc_(b_struct))
arr((60, 80), tc_(b_edge), rad=-0.12)
arr(bc(b_struct), tc_(b_agent))                     # structure -> storyboard
arr(bc(b_agent), tc_(b_asm), label="plan", labfs=7.5)
arr(bc(b_asm), tc_(b_dance))
arr(bc(b_dance), tc_(b_render))
arr(bc(b_render), tc_(b_video))

# LODGE + EDGE motion feed the assembler (curved into its sides)
arr((18, 66.5), (30, 50.5), rad=0.22, label="motion", labfs=7)
arr((82, 66.5), (70, 50.5), rad=-0.22, label="motion", labfs=7)

# raw LODGE / EDGE also rendered as the two reference panels (dashed side channels)
ortho([(5, 70.2), (2.5, 70.2), (2.5, 25.9), (30, 25.9)], color=SIDE_C,
      label="raw LODGE panel", label_pt=(2.5, 48))
ortho([(95, 70.2), (97.5, 70.2), (97.5, 25.9), (70, 25.9)], color=SIDE_C,
      label="raw EDGE panel", label_pt=(97.5, 48), rot=-90)

fig.savefig("story_pipeline_diagram.png", dpi=130, bbox_inches="tight",
            facecolor="white")
print("saved story_pipeline_diagram.png")
