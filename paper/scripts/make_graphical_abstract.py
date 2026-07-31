#!/usr/bin/env python3
"""Generate a publication-style graphical abstract for IntentDiT.

Design goal:
    A clean left-to-right visual story that can be redrawn in Canva:
    inputs -> intent/saliency/text-conditioned diffusion -> generated layout.

The figure uses only synthetic vector shapes, so it is safe for publication upload
and does not depend on benchmark images or third-party assets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "fig"
OUT_PDF = FIG_DIR / "graphical_abstract.pdf"
OUT_PNG = FIG_DIR / "graphical_abstract.png"
OUT_PNG_EXACT = FIG_DIR / "graphical_abstract_3000x1200.png"


C = {
    "ink": "#14213D",
    "muted": "#64748B",
    "line": "#CBD5E1",
    "panel": "#F8FAFC",
    "blue": "#2563EB",
    "blue_soft": "#DBEAFE",
    "green": "#16A34A",
    "green_soft": "#DCFCE7",
    "teal": "#0891B2",
    "teal_soft": "#E0F7FA",
    "orange": "#F59E0B",
    "orange_soft": "#FEF3C7",
    "pink": "#DB2777",
    "pink_soft": "#FCE7F3",
    "violet": "#7C3AED",
    "violet_soft": "#EDE9FE",
    "red": "#DC2626",
    "red_soft": "#FEE2E2",
}


def box(ax, x, y, w, h, fc="white", ec=None, lw=1.8, r=0.025, z=1):
    if ec is None:
        ec = C["line"]
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=12, weight="normal", color=None, ha="center", va="center"):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color or C["ink"],
        family="DejaVu Sans",
    )


def arrow(ax, a, b, color=None, lw=2.6, ms=20, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color or C["blue"],
            connectionstyle=f"arc3,rad={rad}",
            zorder=10,
        )
    )


def poster(ax, x, y, w, h, layout=False, label="Background"):
    """Draw a stylized poster/background."""
    box(ax, x, y, w, h, fc="white", ec=C["line"], lw=1.7, r=0.018, z=2)
    ax.add_patch(Rectangle((x + 0.05 * w, y + 0.05 * h), 0.90 * w, 0.90 * h, color="#ECFEFF", zorder=2))
    ax.add_patch(Rectangle((x + 0.05 * w, y + 0.05 * h), 0.90 * w, 0.28 * h, color="#BAE6FD", zorder=2))
    # salient object
    ax.add_patch(Circle((x + 0.50 * w, y + 0.52 * h), 0.16 * w, color="#FDE68A", zorder=3))
    ax.add_patch(Rectangle((x + 0.43 * w, y + 0.33 * h), 0.17 * w, 0.35 * h, color="#38BDF8", alpha=0.78, zorder=3))
    ax.add_patch(Rectangle((x + 0.12 * w, y + 0.78 * h), 0.18 * w, 0.055 * h, color="#4ADE80", zorder=3))
    if layout:
        # generated elements placed around content
        ax.add_patch(Rectangle((x + 0.12 * w, y + 0.74 * h), 0.30 * w, 0.055 * h, color=C["pink"], alpha=0.9, zorder=4))
        for yy in (0.70, 0.61, 0.52):
            ax.add_patch(Rectangle((x + 0.64 * w, y + yy * h), 0.25 * w, 0.05 * h, color=C["green"], alpha=0.92, zorder=4))
        ax.add_patch(Rectangle((x + 0.60 * w, y + 0.42 * h), 0.32 * w, 0.08 * h, color="#93C5FD", alpha=0.9, zorder=4))
    txt(ax, x + w / 2, y - 0.045, label, size=11, weight="bold")


def mini_map(ax, x, y, w, h, kind):
    fc = {"saliency": C["red_soft"], "intent": C["green_soft"]}[kind]
    ec = {"saliency": C["red"], "intent": C["green"]}[kind]
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.5, r=0.015, z=2)
    if kind == "saliency":
        ax.add_patch(Circle((x + 0.50 * w, y + 0.52 * h), 0.19 * w, color="#FCA5A5", zorder=3))
        ax.add_patch(Rectangle((x + 0.42 * w, y + 0.22 * h), 0.18 * w, 0.50 * h, color="#EF4444", alpha=0.55, zorder=3))
        txt(ax, x + w / 2, y - 0.035, "saliency: avoid", size=10, weight="bold", color=C["red"])
    else:
        for yy, ww in [(0.72, 0.46), (0.58, 0.38), (0.44, 0.34)]:
            ax.add_patch(Rectangle((x + 0.47 * w, y + yy * h), ww * w, 0.07 * h, color=C["green"], alpha=0.8, zorder=3))
        ax.add_patch(Rectangle((x + 0.10 * w, y + 0.74 * h), 0.24 * w, 0.08 * h, color=C["green"], alpha=0.75, zorder=3))
        txt(ax, x + w / 2, y - 0.035, "intent: place", size=10, weight="bold", color=C["green"])


def token_chips(ax, x, y, w, h):
    box(ax, x, y, w, h, fc="white", ec=C["orange"], lw=1.5, r=0.016, z=2)
    txt(ax, x + 0.50 * w, y + 0.58 * h, "Text prompt", size=10.2, weight="bold")
    txt(ax, x + 0.50 * w, y + 0.36 * h, "logo top, 3 texts", size=8.6, color=C["muted"])
    txt(ax, x + w / 2, y - 0.035, "text: control", size=10, weight="bold", color=C["orange"])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Elsevier requests a 500:200 (2.5:1) graphical abstract ratio.
    # The PNG export below is exactly 3000 x 1200 px.
    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    txt(ax, 0.50, 0.94, "Intent-aware diffusion transformer for text-guided poster layout", size=21, weight="bold")

    # Column panels
    box(ax, 0.035, 0.20, 0.23, 0.62, fc=C["panel"], ec=C["line"], r=0.03)
    box(ax, 0.315, 0.20, 0.37, 0.62, fc="#F0F9FF", ec="#BAE6FD", r=0.03)
    box(ax, 0.735, 0.20, 0.23, 0.62, fc=C["panel"], ec=C["line"], r=0.03)

    txt(ax, 0.15, 0.78, "Inputs", size=17, weight="bold")
    txt(ax, 0.50, 0.78, "IntentDiT", size=17, weight="bold")
    txt(ax, 0.85, 0.78, "Generated layout", size=17, weight="bold")

    # Inputs
    poster(ax, 0.070, 0.36, 0.105, 0.31, layout=False, label="image")
    box(ax, 0.182, 0.49, 0.060, 0.065, fc=C["orange_soft"], ec=C["orange"], lw=1.4, r=0.014)
    txt(ax, 0.212, 0.525, "brief", size=10, weight="bold")
    txt(ax, 0.212, 0.455, "“logo top,\n3 text boxes”", size=10)

    # Model: three signals, one denoising core
    mini_map(ax, 0.345, 0.55, 0.095, 0.12, "saliency")
    mini_map(ax, 0.455, 0.55, 0.095, 0.12, "intent")
    token_chips(ax, 0.558, 0.55, 0.108, 0.12)

    box(ax, 0.370, 0.34, 0.260, 0.115, fc="white", ec=C["blue"], lw=2.2, r=0.022)
    txt(ax, 0.500, 0.408, "Diffusion transformer", size=15, weight="bold")
    txt(ax, 0.500, 0.372, "cross-attention + iterative denoising", size=10.5, color=C["muted"])

    txt(ax, 0.50, 0.275, "learns where to place elements and what the prompt requests", size=11, color=C["muted"])

    # Output
    poster(ax, 0.795, 0.405, 0.110, 0.30, layout=True, label="")
    txt(ax, 0.850, 0.365, "prompt-conditioned poster", size=11, weight="bold")
    box(ax, 0.775, 0.285, 0.150, 0.052, fc=C["green_soft"], ec=C["green"], lw=1.4, r=0.014)
    txt(ax, 0.850, 0.314, "density / structure", size=11.5, weight="bold", color="#166534")
    box(ax, 0.775, 0.220, 0.150, 0.052, fc=C["violet_soft"], ec="#818CF8", lw=1.4, r=0.014)
    txt(ax, 0.850, 0.249, "count + spatial cues", size=11.5, weight="bold", color="#3730A3")

    # Main arrows
    arrow(ax, (0.265, 0.51), (0.315, 0.51), color=C["blue"])
    arrow(ax, (0.685, 0.51), (0.735, 0.51), color=C["blue"])
    arrow(ax, (0.392, 0.55), (0.442, 0.455), color=C["red"], lw=1.8, ms=15)
    arrow(ax, (0.502, 0.55), (0.500, 0.455), color=C["green"], lw=1.8, ms=15)
    arrow(ax, (0.612, 0.55), (0.558, 0.455), color=C["orange"], lw=1.8, ms=15)

    # Evidence strip
    box(ax, 0.195, 0.075, 0.19, 0.075, fc=C["green_soft"], ec=C["green"], lw=1.6, r=0.018)
    txt(ax, 0.290, 0.115, "density-aware generation", size=12, weight="bold", color="#166534")
    txt(ax, 0.290, 0.090, "utilization / structure trade-off", size=9.2, color="#166534")

    box(ax, 0.405, 0.075, 0.19, 0.075, fc=C["violet_soft"], ec=C["violet"], lw=1.6, r=0.018)
    txt(ax, 0.500, 0.115, "free-form control", size=12, weight="bold", color="#4C1D95")
    txt(ax, 0.500, 0.090, "higher count overlap; CGL spatial", size=9.2, color="#4C1D95")

    box(ax, 0.615, 0.075, 0.19, 0.075, fc=C["pink_soft"], ec=C["pink"], lw=1.6, r=0.018)
    txt(ax, 0.710, 0.115, "human preference", size=12, weight="bold", color="#9D174D")
    txt(ax, 0.710, 0.090, "85.8% quality / 91.3% instruction", size=9.2, color="#9D174D")

    # Export as a separate graphical-abstract file, not embedded in main.tex.
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PNG_EXACT, dpi=200)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PNG_EXACT}")


if __name__ == "__main__":
    main()
