from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compares several LSTM Autoencoder runs on the same test setup."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in form LABEL=RUN_DIR. Use multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="Wykresy\\lstm_reports\\window_length_comparison",
        help="Output directory for comparison plots.",
    )
    return parser.parse_args()


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Run spec must use LABEL=RUN_DIR format: {spec}")
    label, run_dir = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Run label cannot be empty: {spec}")
    return label, Path(run_dir.strip())


def _load_scores(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            split = raw["split"]
            if split not in {"test_normal", "test_anomaly"}:
                continue
            rows.append(
                {
                    **raw,
                    "score": float(raw["reconstruction_error"]),
                    "label": 1 if split == "test_anomaly" else 0,
                    "is_anomaly_bool": raw["is_anomaly"].strip().lower() == "true",
                }
            )
    return rows


def _binary_metrics_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    preds = scores > threshold
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
    }


def _compute_roc_pr(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return {
            "thresholds": np.zeros((0,), dtype=np.float64),
            "roc_fpr": np.zeros((0,), dtype=np.float64),
            "roc_tpr": np.zeros((0,), dtype=np.float64),
            "pr_precision": np.zeros((0,), dtype=np.float64),
            "pr_recall": np.zeros((0,), dtype=np.float64),
            "roc_auc": None,
            "average_precision": None,
        }

    thresholds = np.unique(scores)[::-1]
    roc_fpr: list[float] = [0.0]
    roc_tpr: list[float] = [0.0]
    pr_precision: list[float] = [1.0]
    pr_recall: list[float] = [0.0]

    for threshold in thresholds:
        metrics = _binary_metrics_at_threshold(scores, labels, float(threshold))
        roc_fpr.append(metrics["fpr"])
        roc_tpr.append(metrics["recall"])
        pr_precision.append(metrics["precision"])
        pr_recall.append(metrics["recall"])

    roc_fpr.append(1.0)
    roc_tpr.append(1.0)
    pr_precision.append(float(np.mean(labels)))
    pr_recall.append(1.0)

    roc_fpr_arr = np.asarray(roc_fpr, dtype=np.float64)
    roc_tpr_arr = np.asarray(roc_tpr, dtype=np.float64)
    pr_precision_arr = np.asarray(pr_precision, dtype=np.float64)
    pr_recall_arr = np.asarray(pr_recall, dtype=np.float64)

    roc_order = np.argsort(roc_fpr_arr)
    pr_order = np.argsort(pr_recall_arr)
    return {
        "thresholds": thresholds,
        "roc_fpr": roc_fpr_arr,
        "roc_tpr": roc_tpr_arr,
        "pr_precision": pr_precision_arr,
        "pr_recall": pr_recall_arr,
        "roc_auc": float(np.trapezoid(roc_tpr_arr[roc_order], roc_fpr_arr[roc_order])),
        "average_precision": float(
            np.trapezoid(pr_precision_arr[pr_order], pr_recall_arr[pr_order])
        ),
    }


def _load_run(label: str, run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    rows = _load_scores(run_dir / "scores_by_window.csv")
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int32)
    threshold = float(metrics["threshold"]["reconstruction_error_threshold"])
    binary = _binary_metrics_at_threshold(scores, labels, threshold)
    roc_pr = _compute_roc_pr(scores, labels)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "rows": rows,
        "scores": scores,
        "labels": labels,
        "threshold": threshold,
        "binary": binary,
        "roc_pr": roc_pr,
    }


def _summarize_sessions(run: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run["rows"]:
        grouped[row["session_key"]].append(row)

    output: list[dict[str, Any]] = []
    for session_key, rows in grouped.items():
        scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
        anomaly_mask = np.asarray([row["is_anomaly_bool"] for row in rows], dtype=bool)
        output.append(
            {
                "window_label": run["label"],
                "session_key": session_key,
                "split": rows[0]["split"],
                "session_state": rows[0]["session_state"],
                "count": int(scores.size),
                "mean_error": float(scores.mean()),
                "p95_error": float(np.quantile(scores, 0.95)),
                "max_error": float(scores.max()),
                "above_threshold_ratio": float(anomaly_mask.mean()),
                "above_threshold_count": int(anomaly_mask.sum()),
            }
        )
    return sorted(output, key=lambda row: (row["split"], row["session_key"], row["window_label"]))


def _save(fig: Any, png_path: Path, svg_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)


def _plot_metric_bars(runs: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["precision", "recall", "specificity", "f1"]
    labels = [run["label"] for run in runs]
    x = np.arange(len(metrics), dtype=np.float64)
    width = 0.22
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, run in enumerate(runs):
        values = [run["binary"][metric] for metric in metrics]
        offset = (idx - (len(runs) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=run["label"], color=colors[idx % len(colors)])

    ax.set_title("Window length comparison - metrics at selected threshold")
    ax.set_ylabel("Value")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "Specificity", "F1"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "01_binary_metrics.png", output_dir / "01_binary_metrics.svg")
    plt.close(fig)


def _plot_auc_bars(runs: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [run["label"] for run in runs]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.32
    roc = [run["roc_pr"]["roc_auc"] or 0.0 for run in runs]
    ap = [run["roc_pr"]["average_precision"] or 0.0 for run in runs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2.0, roc, width=width, label="ROC AUC", color="#0891b2")
    ax.bar(x + width / 2.0, ap, width=width, label="Average precision", color="#ea580c")
    ax.set_title("Window length comparison - ranking metrics")
    ax.set_ylabel("Value")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "02_auc_ap_metrics.png", output_dir / "02_auc_ap_metrics.svg")
    plt.close(fig)


def _plot_roc_pr(runs: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for idx, run in enumerate(runs):
        roc_pr = run["roc_pr"]
        color = colors[idx % len(colors)]
        axes[0].plot(
            roc_pr["roc_fpr"],
            roc_pr["roc_tpr"],
            linewidth=2.4,
            color=color,
            label=f"{run['label']} (AUC={roc_pr['roc_auc']:.3f})",
        )
        axes[1].plot(
            roc_pr["pr_recall"],
            roc_pr["pr_precision"],
            linewidth=2.4,
            color=color,
            label=f"{run['label']} (AP={roc_pr['average_precision']:.3f})",
        )

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.3)
    axes[0].set_title("ROC curves")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Precision-Recall curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    _save(fig, output_dir / "03_roc_pr_curves.png", output_dir / "03_roc_pr_curves.svg")
    plt.close(fig)


def _plot_test_session_ratios(session_rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sessions = sorted({row["session_key"] for row in session_rows})
    labels = list(dict.fromkeys(row["window_label"] for row in session_rows))
    label_order = {label: idx for idx, label in enumerate(labels)}
    sessions = sorted(sessions)
    x = np.arange(len(sessions), dtype=np.float64)
    width = min(0.22, 0.8 / max(len(labels), 1))
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed"]

    values_by_key = {
        (row["session_key"], row["window_label"]): row["above_threshold_ratio"] * 100.0
        for row in session_rows
    }

    fig_width = max(12, len(sessions) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    for label in labels:
        idx = label_order[label]
        offset = (idx - (len(labels) - 1) / 2.0) * width
        values = [values_by_key.get((session, label), 0.0) for session in sessions]
        ax.bar(x + offset, values, width=width, label=label, color=colors[idx % len(colors)])

    ax.set_title("Test sessions - windows above threshold")
    ax.set_ylabel("Above threshold [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(
        fig,
        output_dir / "04_test_session_above_threshold.png",
        output_dir / "04_test_session_above_threshold.svg",
    )
    plt.close(fig)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    summary_rows: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Window length comparison",
        "",
        "## Runs",
        "",
        "| Window | Best epoch | Val loss | Threshold | Precision | Recall | Specificity | F1 | ROC AUC | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['window_label']} | {row['best_epoch']} | {row['best_val_loss']:.6f} | "
            f"{row['threshold']:.6f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['specificity']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | "
            f"{row['average_precision']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Test sessions",
            "",
            "| Window | Session | Split | State | Count | Mean | P95 | Max | Above threshold |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in session_rows:
        lines.append(
            f"| {row['window_label']} | {row['session_key']} | {row['split']} | "
            f"{row['session_state']} | {row['count']} | {row['mean_error']:.4f} | "
            f"{row['p95_error']:.4f} | {row['max_error']:.4f} | "
            f"{row['above_threshold_ratio']:.4f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [_load_run(*_parse_run_spec(spec)) for spec in args.run]
    summary_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []

    for run in runs:
        metrics = run["metrics"]
        binary = run["binary"]
        roc_pr = run["roc_pr"]
        summary_rows.append(
            {
                "window_label": run["label"],
                "run_dir": run["run_dir"],
                "best_epoch": int(metrics["best_epoch"]),
                "best_val_loss": float(metrics["best_val_loss"]),
                "threshold": float(run["threshold"]),
                "precision": float(binary["precision"]),
                "recall": float(binary["recall"]),
                "specificity": float(binary["specificity"]),
                "fpr": float(binary["fpr"]),
                "f1": float(binary["f1"]),
                "roc_auc": float(roc_pr["roc_auc"] or 0.0),
                "average_precision": float(roc_pr["average_precision"] or 0.0),
                "test_normal_count": int(metrics["split_metrics"]["test_normal"]["count"]),
                "test_anomaly_count": int(metrics["split_metrics"]["test_anomaly"]["count"]),
            }
        )
        session_rows.extend(_summarize_sessions(run))

    _plot_metric_bars(runs, output_dir)
    _plot_auc_bars(runs, output_dir)
    _plot_roc_pr(runs, output_dir)
    _plot_test_session_ratios(session_rows, output_dir)

    _write_csv(summary_rows, output_dir / "comparison_summary.csv")
    _write_csv(session_rows, output_dir / "session_comparison.csv")
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(
            {
                "runs": summary_rows,
                "test_sessions": session_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(summary_rows, session_rows, output_dir / "REPORT.md")
    print(f"Saved comparison report to: {output_dir}")
    print(json.dumps({"runs": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
