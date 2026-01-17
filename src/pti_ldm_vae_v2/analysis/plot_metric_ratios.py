from __future__ import annotations

import argparse
import json
from pathlib import Path

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting metric ratios.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Plot metric ratios after LDM sampling.")
    parser.add_argument(
        "--metric",
        default=None,
        help="Optional single metric name (e.g., height_0). If omitted, plot all metrics in a grid.",
    )
    parser.add_argument(
        "--y-mode",
        choices=["ratio", "diff"],
        default="ratio",
        help="Y-axis mode: ratio (pred/edente) or diff (pred - edente).",
    )
    parser.add_argument("--y-min", type=float, default=None, help="Optional fixed minimum for the y-axis.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional fixed maximum for the y-axis.")
    parser.add_argument("--edente-metrics", required=True, type=Path, help="Path to edente metrics JSON.")
    parser.add_argument("--dente-metrics", required=True, type=Path, help="Path to dente metrics JSON.")
    parser.add_argument("--pred-metrics", required=True, type=Path, help="Path to predicted edente metrics JSON.")
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Optional output HTML path (defaults next to pred metrics).",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    """Load metrics from a JSON file.

    Args:
        path (Path): JSON path.

    Returns:
        dict[str, dict[str, float]]: Mapping from filename to metric values.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected metrics format in {path}")
    return data


def infer_metrics(
    edente: dict[str, dict[str, float]],
    dente: dict[str, dict[str, float]],
    pred: dict[str, dict[str, float]],
) -> list[str]:
    """Infer shared metric names from the first entries.

    Args:
        edente (dict[str, dict[str, float]]): Ground-truth edente metrics.
        dente (dict[str, dict[str, float]]): Ground-truth dente metrics.
        pred (dict[str, dict[str, float]]): Predicted edente metrics.

    Returns:
        list[str]: Sorted list of metric names.
    """
    def first_metrics(data: dict[str, dict[str, float]]) -> set[str]:
        for values in data.values():
            return set(values.keys())
        return set()

    metrics = first_metrics(edente) & first_metrics(dente) & first_metrics(pred)

    def sort_key(name: str) -> tuple[int, int | str]:
        if name == "height_0":
            return (0, 0)
        if name.startswith("width_"):
            suffix = name.split("_", maxsplit=1)[-1]
            return (1, int(suffix)) if suffix.isdigit() else (1, suffix)
        return (2, name)

    return sorted(metrics, key=sort_key)


def compute_ratios(
    metric: str,
    edente: dict[str, dict[str, float]],
    dente: dict[str, dict[str, float]],
    pred: dict[str, dict[str, float]],
    *,
    y_mode: str,
) -> tuple[list[float], list[float], list[str], dict[str, int]]:
    """Compute ratio pairs for one metric.

    Args:
        metric (str): Metric name to compare.
        edente (dict[str, dict[str, float]]): Ground-truth edente metrics.
        dente (dict[str, dict[str, float]]): Ground-truth dente metrics.
        pred (dict[str, dict[str, float]]): Predicted edente metrics.

    Returns:
        tuple[list[float], list[float], list[str], dict[str, int]]:
            x values, y values, filenames, and counters.
    """
    names = sorted(set(edente) & set(dente) & set(pred))
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    counters = {"missing_metric": 0, "zero_division": 0, "used": 0}

    for name in names:
        ed_val = edente[name].get(metric)
        de_val = dente[name].get(metric)
        pr_val = pred[name].get(metric)
        if ed_val is None or de_val is None or pr_val is None:
            counters["missing_metric"] += 1
            continue
        if float(de_val) == 0.0:
            counters["zero_division"] += 1
            continue
        if y_mode == "ratio" and float(ed_val) == 0.0:
            counters["zero_division"] += 1
            continue
        xs.append(float(ed_val) / float(de_val))
        if y_mode == "ratio":
            ys.append(float(pr_val) / float(ed_val))
        else:
            ys.append(float(pr_val) - float(ed_val))
        labels.append(name)
        counters["used"] += 1

    return xs, ys, labels, counters


def build_single_scatter(
    metric: str,
    xs: list[float],
    ys: list[float],
    labels: list[str],
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    y_mode: str,
) -> go.Figure:
    """Create a single scatter plot figure.

    Args:
        metric (str): Metric name.
        xs (list[float]): x-axis values (edente/dente).
        ys (list[float]): y-axis values (pred/edente).
        labels (list[str]): Filenames for hover tooltips.

    Returns:
        go.Figure: Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            text=labels,
            marker={"symbol": "circle-open"},
            hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_annotation(
        x=0.5,
        y=0.86,
        xref="x domain",
        yref="y domain",
        text=f"<b>{metric}</b>",
        showarrow=False,
        font={"size": 16},
    )
    fig.update_layout(template="plotly_white")
    fig.update_xaxes(range=list(x_range))
    fig.update_yaxes(range=list(y_range))
    return fig


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    edente_metrics = load_metrics(args.edente_metrics)
    dente_metrics = load_metrics(args.dente_metrics)
    pred_metrics = load_metrics(args.pred_metrics)

    metrics = [args.metric] if args.metric else infer_metrics(edente_metrics, dente_metrics, pred_metrics)
    if not metrics:
        raise ValueError("No shared metrics found across inputs.")

    per_metric: list[tuple[str, list[float], list[float], list[str], dict[str, int]]] = []
    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")

    for metric in metrics:
        xs, ys, labels, counters = compute_ratios(
            metric=metric,
            edente=edente_metrics,
            dente=dente_metrics,
            pred=pred_metrics,
            y_mode=args.y_mode,
        )
        if not xs:
            print(f"[WARN] No valid points for metric '{metric}'. Skipping.")
            continue
        per_metric.append((metric, xs, ys, labels, counters))
        x_min = min(x_min, min(xs))
        x_max = max(x_max, max(xs))
        y_min = min(y_min, min(ys))
        y_max = max(y_max, max(ys))

    if not per_metric:
        raise ValueError("No valid points found. Check metric names and inputs.")

    x_range = (0.99 * x_min, 1.01 * x_max)
    y_lower = 0.99 * y_min if args.y_min is None else float(args.y_min)
    y_upper = 1.01 * y_max if args.y_max is None else float(args.y_max)
    if y_lower >= y_upper:
        raise ValueError("y-min must be smaller than y-max.")
    y_range = (y_lower, y_upper)

    output_path = args.output_html
    if output_path is None:
        suffix = per_metric[0][0] if len(per_metric) == 1 else "grid"
        output_path = args.pred_metrics.with_name(f"metric_scatter_{suffix}.html")

    if len(per_metric) == 1:
        metric, xs, ys, labels, _ = per_metric[0]
        fig = build_single_scatter(metric, xs, ys, labels, x_range=x_range, y_range=y_range, y_mode=args.y_mode)
    else:
        if len(per_metric) > 6:
            raise ValueError("Only up to 6 metrics are supported in grid mode.")
        fig = make_subplots(
            rows=2,
            cols=3,
            vertical_spacing=0.03,
            horizontal_spacing=0.02,
        )
        for idx, (metric, xs, ys, labels, _) in enumerate(per_metric):
            row = idx // 3 + 1
            col = idx % 3 + 1
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    text=labels,
                    marker={"symbol": "circle-open"},
                    hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.add_annotation(
                x=0.5,
                y=0.86,
                xref="x domain",
                yref="y domain",
                text=f"<b>{metric}</b>",
                showarrow=False,
                font={"size": 14},
                row=row,
                col=col,
            )

        for idx in range(len(per_metric), 6):
            row = idx // 3 + 1
            col = idx % 3 + 1
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)

        fig.update_layout(
            template="plotly_white",
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
        fig.update_xaxes(range=list(x_range))
        fig.update_yaxes(range=list(y_range))

    fig.write_html(str(output_path), include_plotlyjs="cdn")

    summary = {
        "output": str(output_path),
        "metrics": [item[0] for item in per_metric],
        "points": sum(item[4]["used"] for item in per_metric),
        "skipped_missing_metric": sum(item[4]["missing_metric"] for item in per_metric),
        "skipped_zero_division": sum(item[4]["zero_division"] for item in per_metric),
    }
    print("Saved scatter plot", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
