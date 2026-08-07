"""Render combined use-case diagram + specification tables as one PNG."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch
from matplotlib.table import Table

ACTORS = [
    ("Researcher", "Primary", "Run pipeline, configure flags, interpret results"),
    ("Google Earth Engine", "External", "Sentinel-1/2 flood detection"),
    ("Colab SITS-VAE", "External", "Patch scoring inference"),
    ("GDELT BigQuery", "External", "Multilingual article counts (JSON)"),
    ("EM-DAT", "Data", "Event list, deaths, metadata"),
    ("Natural Earth", "Data", "State choropleth boundaries"),
]

USE_CASES = [
    ("UC-01", "Build Event Data", "Researcher, EM-DAT", "Convert EM-DAT floods to analysis events", "events.csv", "build_events_emdat.py", "legacy"),
    ("UC-02", "Satellite Detection", "Researcher, GEE, Colab", "Flood extent + SITS patches per event", "flood_extent.csv", "satellite.py", "optional"),
    ("UC-03", "Merge Flood Areas", "Researcher", "Fuse SITS / NDWI / SAR into one area", "flood_combined.csv", "merge_results.py", "core"),
    ("UC-04", "Pop. / Severity", "Researcher", "Estimate exposed population from flood area", "severity_raw.csv", "compute_population.py", "core"),
    ("UC-05", "Compute PSS", "Researcher", "Physical Severity Score (area + population)", "pss_results.csv", "compute_pss.py", "core"),
    ("UC-06", "Compute MSS", "Researcher, GDELT", "Media Salience Score (3 components, AHP)", "mss_results.csv", "compute_mss.py", "core"),
    ("UC-07", "Expected Coverage", "Researcher, EM-DAT", "NegBin model + log_ratio discrepancy", "expected_coverage.csv", "compute_expected_coverage.py", "core"),
    ("UC-08", "Visualization", "Researcher, Natural Earth", "Plots + pipeline report", "plot*.png", "visualize.py", "core"),
    ("UC-09", "Run Pipeline", "Researcher", "End-to-end cached rebuild via shell script", "data/, outputs/", "run_pipeline.sh", "orchestrator"),
    ("UC-10", "Overdispersion Test", "Researcher", "Validate NegBin vs Poisson / OLS", "overdispersion_report.md", "test_negbin_overdispersion.py", "test"),
    ("UC-11", "MSS Weight Compare", "Researcher", "AHP vs PCA / EWM / Equal sensitivity", "mss_weight_report.md", "test_mss_weight_methods.py", "test"),
]

MODE_COLOR = {
    "core": "#D6EAF8",
    "orchestrator": "#D5F5E3",
    "optional": "#FDEBD0",
    "test": "#E8DAEF",
    "legacy": "#F2F3F4",
}


def draw_ellipse(ax, cx, cy, w, h, label, sub="", style="core", dashed=False):
    fc = MODE_COLOR.get(style, "#FFFFFF")
    ec = "#888888" if dashed else "#333333"
    ls = (0, (4, 3)) if dashed else "solid"
    e = Ellipse((cx, cy), w, h, facecolor=fc, edgecolor=ec, linewidth=1.2, linestyle=ls, zorder=3)
    ax.add_patch(e)
    ax.text(cx, cy + 0.015, label, ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=4)
    if sub:
        ax.text(cx, cy - 0.045, sub, ha="center", va="center", fontsize=6.2, zorder=4)


def draw_actor_box(ax, x, y, w, h, title, sub=""):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor="#F5F5F5", edgecolor="#333", linewidth=1, zorder=3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=5.8, zorder=4)


def arrow(ax, x1, y1, x2, y2, dashed=False):
    style = "->,head_width=0.08,head_length=0.12"
    color = "#888888" if dashed else "#555555"
    ls = (0, (4, 3)) if dashed else "solid"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=color, linestyle=ls, zorder=2))


def draw_diagram(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.97, "Use Case Diagram", ha="center", va="top", fontsize=11, fontweight="bold")

    # system boundary
    boundary = FancyBboxPatch((0.14, 0.08), 0.72, 0.82, boxstyle="round,pad=0.01", facecolor="#FAFAFA", edgecolor="#333", linewidth=1.5, zorder=1)
    ax.add_patch(boundary)
    ax.text(0.5, 0.86, "CVND Pipeline System", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # use cases
    draw_ellipse(ax, 0.28, 0.76, 0.11, 0.07, "UC-09", "Run Pipeline", "orchestrator")
    draw_ellipse(ax, 0.48, 0.76, 0.11, 0.07, "UC-02", "Satellite", "optional", dashed=True)
    draw_ellipse(ax, 0.20, 0.60, 0.10, 0.06, "UC-03", "Merge Areas", "core")
    draw_ellipse(ax, 0.33, 0.60, 0.10, 0.06, "UC-04", "Pop/Severity", "core")
    draw_ellipse(ax, 0.46, 0.60, 0.09, 0.06, "UC-05", "PSS", "core")
    draw_ellipse(ax, 0.58, 0.60, 0.09, 0.06, "UC-06", "MSS", "core")
    draw_ellipse(ax, 0.42, 0.44, 0.11, 0.06, "UC-07", "Expected Cov.", "core")
    draw_ellipse(ax, 0.58, 0.44, 0.10, 0.06, "UC-08", "Visualize", "core")
    draw_ellipse(ax, 0.30, 0.28, 0.11, 0.06, "UC-10", "Overdispersion", "test")
    draw_ellipse(ax, 0.52, 0.28, 0.11, 0.06, "UC-11", "MSS Weights", "test")
    draw_ellipse(ax, 0.70, 0.76, 0.10, 0.06, "UC-01", "Events", "legacy", dashed=True)

    # researcher
    ax.add_patch(Ellipse((0.06, 0.55), 0.07, 0.09, facecolor="#E8F0FE", edgecolor="#333", linewidth=1.2, zorder=3))
    ax.text(0.06, 0.565, "Researcher", ha="center", va="center", fontsize=6.5, fontweight="bold")
    ax.text(0.06, 0.525, "(Primary)", ha="center", va="center", fontsize=5.8)

    # external actors
    draw_actor_box(ax, 0.88, 0.74, 0.10, 0.06, "Google Earth", "Engine")
    draw_actor_box(ax, 0.88, 0.64, 0.10, 0.06, "Colab", "SITS-VAE")
    draw_actor_box(ax, 0.88, 0.54, 0.10, 0.06, "GDELT", "BigQuery")
    draw_actor_box(ax, 0.88, 0.44, 0.10, 0.06, "EM-DAT", "")
    draw_actor_box(ax, 0.88, 0.34, 0.10, 0.06, "Natural Earth", "")

    # associations
    arrow(ax, 0.10, 0.58, 0.22, 0.76)
    arrow(ax, 0.10, 0.52, 0.25, 0.30)
    arrow(ax, 0.10, 0.50, 0.47, 0.30)
    arrow(ax, 0.10, 0.54, 0.53, 0.44)

    # include / extend from UC-09
    for tx, ty in [(0.20, 0.64), (0.33, 0.64), (0.46, 0.64), (0.58, 0.64), (0.42, 0.50), (0.58, 0.50)]:
        arrow(ax, 0.28, 0.72, tx, ty)
    arrow(ax, 0.34, 0.76, 0.43, 0.76, dashed=True)

    # pipeline chain
    arrow(ax, 0.25, 0.60, 0.28, 0.60)
    arrow(ax, 0.38, 0.60, 0.41, 0.60)
    arrow(ax, 0.51, 0.60, 0.535, 0.60)
    arrow(ax, 0.46, 0.56, 0.44, 0.48)
    arrow(ax, 0.58, 0.56, 0.46, 0.47)
    arrow(ax, 0.52, 0.44, 0.53, 0.44)

    # external links
    arrow(ax, 0.88, 0.77, 0.53, 0.76, dashed=True)
    arrow(ax, 0.88, 0.67, 0.53, 0.74, dashed=True)
    arrow(ax, 0.88, 0.67, 0.25, 0.62, dashed=True)
    arrow(ax, 0.88, 0.57, 0.625, 0.60, dashed=True)
    arrow(ax, 0.88, 0.47, 0.475, 0.46, dashed=True)
    arrow(ax, 0.88, 0.37, 0.63, 0.44, dashed=True)
    arrow(ax, 0.88, 0.77, 0.65, 0.76, dashed=True)

    ax.text(0.36, 0.68, "<<include>>", fontsize=6, color="#555", style="italic")
    ax.text(0.40, 0.79, "<<extend>>", fontsize=6, color="#888", style="italic")


def add_table(ax, title, headers, rows, col_widths, fontsize=6.2, header_color="#E8E8E8"):
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left", pad=6)
    table = ax.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="left", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(fontweight="bold", fontsize=fontsize)
        elif r % 2 == 0:
            cell.set_facecolor("#FAFAFA")


def main():
    fig = plt.figure(figsize=(16, 22), dpi=150)
    fig.patch.set_facecolor("white")
    fig.suptitle("Figure 3-1. CVND System Analysis — Use Case Diagram & Specifications", fontsize=14, fontweight="bold", y=0.995)

    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 0.42, 1.35], hspace=0.28, wspace=0.12, left=0.04, right=0.96, top=0.965, bottom=0.02)

    ax_diag = fig.add_subplot(gs[0, :])
    draw_diagram(ax_diag)

    ax_actors = fig.add_subplot(gs[1, :])
    actor_rows = [[a, t, r] for a, t, r in ACTORS]
    add_table(ax_actors, "Table 3-1. Actors", ["Actor", "Type", "Role"], actor_rows, [0.18, 0.12, 0.70], fontsize=6.5)

    ax_spec = fig.add_subplot(gs[2, :])
    spec_rows = [[uc[0], uc[1], uc[2], uc[3], uc[4], uc[5]] for uc in USE_CASES]
    add_table(
        ax_spec,
        "Table 3-2. Use Case Specifications",
        ["ID", "Name", "Actor(s)", "Purpose", "Output", "Module"],
        spec_rows,
        [0.07, 0.11, 0.14, 0.30, 0.14, 0.24],
        fontsize=6.0,
    )

    # legend bar under diagram
    legend_y = 0.56
    ax_leg = fig.add_axes([0.04, legend_y, 0.92, 0.025])
    ax_leg.axis("off")
    patches = [
        mpatches.Patch(facecolor=MODE_COLOR["core"], edgecolor="#333", label="Core"),
        mpatches.Patch(facecolor=MODE_COLOR["orchestrator"], edgecolor="#333", label="Orchestrator"),
        mpatches.Patch(facecolor=MODE_COLOR["optional"], edgecolor="#888", label="Optional"),
        mpatches.Patch(facecolor=MODE_COLOR["test"], edgecolor="#666", label="Validation"),
        mpatches.Patch(facecolor=MODE_COLOR["legacy"], edgecolor="#888", label="Legacy"),
    ]
    ax_leg.legend(handles=patches, loc="center", ncol=5, fontsize=7, frameon=False)

    out = "/Users/rokpolar/Documents/CU_CDI/CVND/docs/figures/use_case_diagram.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
