"""Render the factorial summary figures with Matplotlib."""

# Keep the backend selection before importing pyplot; this intentionally splits
# the Matplotlib import block.
# ruff: noqa: I001

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "analysis" / "results.csv"
FIGURES = ROOT / "figures"
HORIZONS = [1, 5, 20, 44]
SERIES = [
    ("correct", "onestep", "Correct gravity, one-step", "#2563eb", "-"),
    ("constant", "onestep", "Constant gravity, one-step", "#f97316", "--"),
    ("correct", "rollout", "Correct gravity, rollout", "#16a34a", "-"),
    ("constant", "rollout", "Constant gravity, rollout", "#9333ea", "--"),
]


def read_rows() -> list[dict[str, float | str]]:
    with DATA_PATH.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "horizon": int(row["horizon"]),
                    "gravity_condition": row["gravity_condition"],
                    "training_objective": row["training_objective"],
                    "target_mse": float(row["target_mse"]),
                    "position_l2": float(row["position_l2"]),
                }
            )
        return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.edgecolor": "#6b7280",
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#d9dee5",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "svg.fonttype": "none",
            "svg.hashsalt": "sg-jepa-gravity-rollout-factorial",
            "pdf.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, stem: str, title: str) -> None:
    svg_metadata = {"Creator": "analysis/plot_results.py", "Title": title, "Date": None}
    pdf_metadata = {
        "Creator": "analysis/plot_results.py",
        "Title": title,
        "CreationDate": None,
        "ModDate": None,
    }
    svg_path = FIGURES / f"{stem}.svg"
    figure.savefig(svg_path, bbox_inches="tight", metadata=svg_metadata)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", metadata=pdf_metadata)
    figure.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def index_rows(rows: list[dict[str, float | str]]) -> dict[tuple[str, str], dict[int, dict[str, float | str]]]:
    indexed: dict[tuple[str, str], dict[int, dict[str, float | str]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["gravity_condition"]), str(row["training_objective"]))
        indexed[key][int(row["horizon"])] = row
    return indexed


def make_error_figure(rows: list[dict[str, float | str]]) -> None:
    indexed = index_rows(rows)
    figure = plt.figure(figsize=(11, 6.3), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=[4.8, 1.0])
    axes = [figure.add_subplot(grid[0, column]) for column in range(2)]
    legend_axis = figure.add_subplot(grid[1, :])
    legend_axis.axis("off")

    figure.suptitle("Autoregressive prediction error", fontsize=19)
    for axis, metric, title in zip(
        axes,
        ("target_mse", "position_l2"),
        ("Normalized target MSE", "Position L2"),
        strict=True,
    ):
        for gravity, objective, label, color, linestyle in SERIES:
            values = [indexed[(gravity, objective)][horizon][metric] for horizon in HORIZONS]
            axis.plot(
                HORIZONS,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                marker="o",
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=label,
            )
        axis.set_title(title, pad=10)
        axis.set_xlabel("prediction horizon", labelpad=8)
        axis.set_ylabel("error", labelpad=8)
        axis.set_xticks(HORIZONS)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)

    legend_series = [SERIES[0], SERIES[2], SERIES[1], SERIES[3]]
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=label,
        )
        for _, _, label, color, linestyle in legend_series
    ]
    legend_axis.legend(handles=handles, loc="center", ncol=2, frameon=False, handlelength=2.4)
    save_figure(figure, "autoregressive-error", "Autoregressive prediction error")


def make_effects_figure(rows: list[dict[str, float | str]]) -> None:
    indexed = index_rows(rows)
    metrics = [("target_mse", "MSE"), ("position_l2", "position")]
    groups = [("Correct gravity", "correct", "#16a34a"), ("Constant gravity", "constant", "#9333ea")]
    positions = [0, 1, 3, 4]
    values: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    for group_label, gravity, color in groups:
        for metric, metric_label in metrics:
            one_step = float(indexed[(gravity, "onestep")][20][metric])
            rollout = float(indexed[(gravity, "rollout")][20][metric])
            values.append((one_step - rollout) / one_step * 100)
            colors.append(color)
            labels.append(f"{metric_label}\n{group_label}")

    figure, axis = plt.subplots(figsize=(9.8, 5.3), layout="constrained")
    figure.suptitle("Horizon-20 rollout effect", fontsize=19)
    bars = axis.bar(positions, values, width=0.72, color=colors, alpha=0.9)
    axis.bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=6, fontsize=12)
    axis.set_xticks(positions, labels)
    axis.set_xlim(-0.65, 4.65)
    axis.set_ylim(0, 80)
    axis.set_ylabel("reduction", labelpad=10)
    axis.set_xlabel("relative reduction from one-step to rollout training", labelpad=12)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    save_figure(figure, "horizon-20-effects", "Horizon-20 rollout effect")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_style()
    rows = read_rows()
    make_error_figure(rows)
    make_effects_figure(rows)


if __name__ == "__main__":
    main()
