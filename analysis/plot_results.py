"""Render the factorial summary figures with the Python standard library."""

from __future__ import annotations

import csv
from collections import defaultdict
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "analysis" / "results.csv"
FIGURES = ROOT / "figures"
HORIZONS = [1, 5, 20, 44]
SERIES = [
    ("correct", "onestep", "Correct gravity, one-step", "#2563eb", "solid"),
    ("constant", "onestep", "Constant gravity, one-step", "#f97316", "dash"),
    ("correct", "rollout", "Correct gravity, rollout", "#16a34a", "solid"),
    ("constant", "rollout", "Constant gravity, rollout", "#9333ea", "dash"),
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


def text(x: float, y: float, value: str, *, size: int = 13, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="#1f2937">{escape(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def circle(x: float, y: float, fill: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{fill}" stroke="#ffffff" stroke-width="1.5"/>'


def line_panel(
    rows: list[dict[str, float | str]],
    metric: str,
    title: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    values = [float(row[metric]) for row in rows]
    ymax = max(values) * 1.12
    left = x + 58
    right = x + width - 18
    top = y + 42
    bottom = y + height - 42
    plot_width = right - left
    plot_height = bottom - top
    x_positions = {
        horizon: left + plot_width * i / (len(HORIZONS) - 1)
        for i, horizon in enumerate(HORIZONS)
    }
    parts = [text(x + width / 2, y + 20, title, size=18, anchor="middle", weight="500")]
    parts.append(line(left, top, left, bottom, "#6b7280", 1.2))
    parts.append(line(left, bottom, right, bottom, "#6b7280", 1.2))
    ticks = 4
    for i in range(ticks + 1):
        value = ymax * i / ticks
        y_tick = bottom - plot_height * i / ticks
        parts.append(line(left, y_tick, right, y_tick, "#e5e7eb", 1.0))
        parts.append(text(left - 9, y_tick + 4, f"{value:.1f}", size=13, anchor="end"))
    for horizon, x_tick in x_positions.items():
        parts.append(line(x_tick, bottom, x_tick, bottom + 5, "#6b7280", 1.0))
        parts.append(text(x_tick, bottom + 24, str(horizon), size=13, anchor="middle"))
    parts.append(text(x + 14, top + plot_height / 2, "error", size=14, anchor="middle"))
    parts.append(text(x + width / 2, y + height - 7, "prediction horizon", size=14, anchor="middle"))

    by_series = defaultdict(dict)
    for row in rows:
        key = (row["gravity_condition"], row["training_objective"])
        by_series[key][int(row["horizon"])] = float(row[metric])
    for gravity, objective, _, color, style in SERIES:
        points = []
        for horizon in HORIZONS:
            point_x = x_positions[horizon]
            point_y = bottom - plot_height * by_series[(gravity, objective)][horizon] / ymax
            points.append((point_x, point_y))
        path = " ".join(
            ("M" if i == 0 else "L") + f" {point_x:.1f} {point_y:.1f}"
            for i, (point_x, point_y) in enumerate(points)
        )
        dash = "7 5" if style == "dash" else None
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"{dash_attr}/>' )
        parts.extend(circle(point_x, point_y, color) for point_x, point_y in points)
    return "".join(parts)


def make_error_figure(rows: list[dict[str, float | str]]) -> None:
    width, height = 1100, 610
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Autoregressive error across horizons</title>',
        '<desc id="desc">Normalized target MSE and position L2 for four gravity and training conditions across horizons 1, 5, 20, and 44.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(width / 2, 31, "Autoregressive prediction error", size=22, anchor="middle", weight="500"),
    ]
    parts.append(line_panel(rows, "target_mse", "Normalized target MSE", 20, 52, 515, 415))
    parts.append(line_panel(rows, "position_l2", "Position L2", 565, 52, 515, 415))
    legend_y = 520
    for index, (_, _, label, color, style) in enumerate(SERIES):
        legend_x = 56 + (index % 2) * 500
        current_y = legend_y + (index // 2) * 30
        dash = "7 5" if style == "dash" else None
        parts.append(line(legend_x, current_y - 5, legend_x + 28, current_y - 5, color, 2.5, dash))
        parts.append(circle(legend_x + 14, current_y - 5, color))
        parts.append(text(legend_x + 40, current_y, label, size=14))
    parts.append("</svg>")
    (FIGURES / "autoregressive-error.svg").write_text("\n".join(parts) + "\n")


def make_effects_figure(rows: list[dict[str, float | str]]) -> None:
    by_key = {(row["gravity_condition"], row["training_objective"]): row for row in rows if row["horizon"] == 20}
    metrics = [("target_mse", "MSE reduction", "percent"), ("position_l2", "Position reduction", "percent")]
    width, height = 980, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Horizon-20 rollout effects</title>',
        '<desc id="desc">Percentage reduction from one-step to rollout training under correct and constant gravity at horizon 20.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(width / 2, 32, "Horizon-20 rollout effect", size=22, anchor="middle", weight="500"),
    ]
    chart_left, chart_top, chart_width, chart_height = 90, 76, 800, 294
    chart_bottom = chart_top + chart_height
    max_value = 80
    for i in range(5):
        value = i * 20
        y_tick = chart_bottom - chart_height * value / max_value
        parts.append(line(chart_left, y_tick, chart_left + chart_width, y_tick, "#e5e7eb", 1.0))
        parts.append(text(chart_left - 10, y_tick + 4, f"{value}%", size=13, anchor="end"))
    parts.append(line(chart_left, chart_top, chart_left, chart_bottom, "#6b7280", 1.2))
    parts.append(line(chart_left, chart_bottom, chart_left + chart_width, chart_bottom, "#6b7280", 1.2))
    groups = [("Correct gravity", "correct", "#16a34a"), ("Constant gravity", "constant", "#9333ea")]
    bar_width = 72
    group_centers = [chart_left + chart_width * 0.31, chart_left + chart_width * 0.69]
    for group_index, (group_label, gravity, color) in enumerate(groups):
        center = group_centers[group_index]
        for metric_index, (metric, label, _) in enumerate(metrics):
            one_step = float(by_key[(gravity, "onestep")][metric])
            rollout = float(by_key[(gravity, "rollout")][metric])
            reduction = (one_step - rollout) / one_step * 100
            bar_x = center - bar_width - 10 + metric_index * (bar_width + 20)
            bar_h = chart_height * reduction / max_value
            bar_y = chart_bottom - bar_h
            parts.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width}" height="{bar_h:.1f}" fill="{color}" opacity="0.88"/>')
            parts.append(text(bar_x + bar_width / 2, bar_y - 10, f"{reduction:.0f}%", size=14, anchor="middle", weight="500"))
            parts.append(text(bar_x + bar_width / 2, chart_bottom + 25, "MSE" if metric == "target_mse" else "position", size=13, anchor="middle"))
        parts.append(text(center, chart_bottom + 62, group_label, size=15, anchor="middle", weight="500"))
    parts.append(text(chart_left + chart_width / 2, height - 31, "relative reduction from one-step to rollout training", size=14, anchor="middle"))
    parts.append(text(chart_left - 56, chart_top + chart_height / 2, "reduction", size=14, anchor="middle"))
    parts.append("</svg>")
    (FIGURES / "horizon-20-effects.svg").write_text("\n".join(parts) + "\n")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    make_error_figure(rows)
    make_effects_figure(rows)


if __name__ == "__main__":
    main()
